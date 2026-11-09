"""t23_comm_bench.py -- 双机 16 卡跨节点传输微基准（RDMA vs TCP 判决数据）。
torchrun --nnodes 2 --nproc-per-node 8 启动（两 worker 各 8 BW1000）。
纯 torch/RCCL：RDMA 臂与 TCP 臂同一份代码，唯一差异是 enable_rdma 的 env
（NCCL_NET_PLUGIN=shca + IBext_v8 vs NET/Socket），判据由外层 grep NET/ 行。

组几何（对齐 16 卡车辆 TP8×DP2 的真实通信类）:
  world16: TP16/vllm 类全世界组 —— AR/RS/AG @ 载荷匹配点（shard 62.9MB,
          t22 round-2 口径 M=49152×5120 bf16）+ 336MB 鲸 + 1MB 时延探针
  intra8 : 节点内 8 卡参考臂（同 shard 载荷）—— 归一化跨节点差值
  dp2    : 跨节点对 (r, r+8) —— TP8×DP2 车辆 DP 梯度面真实形状（336MB 桶类）

铁律（本文件的正确性前提）:
  * dist.new_group 必须 16 rank 全体以相同顺序、相同成员列表调用 —— 两组
    intra 和 8 个 dp2 对全部在所有 rank 上建立，成员资格只决定是否发集合。
  * 非成员不提前 return —— world barrier / 统计 all_reduce 仍要参与，
    否则全世界组互等死锁。
计时律: 只在 compute 流上 CUDA event 计时 + 起点前 world barrier（S43 事件
毒药律的对面：这里没有 copy 流）。跨 rank 报 min/max（S18 跨进程配对律：
聚合在 run 级，不按 iter 配对）。
正确性闸: world 与 dp2 各一次小 AR 求和校验（传输损坏探测器，非计时）。
"""
import argparse
import csv
import datetime
import os

import torch
import torch.distributed as dist

ROWS, COLS = 49152, 5120            # 载荷匹配点: full 503.3MB; /16 与 /8 shard 均 62.9MB
WHALE_ELEMS = 336 * 1024 * 1024 // 2  # 336MiB bf16 的元素数（mappings 鲸/DDO 桶类）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warm", type=int, default=3)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", 0))
    node = rank // 8
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    g_world = dist.group.WORLD

    # 全体 rank 同序建齐所有子组（铁律）
    intra0 = dist.new_group(list(range(0, 8)))
    intra1 = dist.new_group(list(range(8, 16)))
    pairs = [dist.new_group([i, i + 8]) for i in range(8)]
    g_intra = intra0 if node == 0 else intra1
    g_dp2 = pairs[rank % 8]
    rdma_arm = "tcp" if os.environ.get("RDMA_DISABLE", "0") == "1" else "rdma"

    def log(msg):
        if rank == 0:
            print(f"[t23comm] {msg}", flush=True)

    try:
        nccl_ver = ".".join(map(str, torch.cuda.nccl.version()))
    except Exception:
        nccl_ver = "unknown"

    # 正确性闸: 小 AR 求和校验（16 rank 都在自己的组里参与）
    x = torch.arange(256, dtype=torch.float32, device="cuda")
    dist.all_reduce(x, group=g_world)
    ok_w = bool(torch.allclose(
        x.sum(), torch.tensor(256 * world * 255 / 2, device="cuda")))
    y = torch.arange(256, dtype=torch.float32, device="cuda")
    dist.all_reduce(y, group=g_dp2)
    ok_p = bool(torch.allclose(
        y.sum(), torch.tensor(256 * 2 * 255 / 2, device="cuda")))
    log(f"verify world16={'PASS' if ok_w else 'FAIL'} "
        f"dp2={'PASS' if ok_p else 'FAIL'}（FAIL=传输损坏, 须停判）")

    rows = []

    def bench(name, gname, g, gn, fn, nbytes_full, bus_bytes):
        member = dist.get_rank(g) >= 0
        if member:
            for _ in range(args.warm):
                fn()
        torch.cuda.synchronize()
        dist.barrier()                     # 16 rank 全体到齐（含非成员）
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        if member:
            for _ in range(args.iters):
                fn()
        e.record()
        torch.cuda.synchronize()
        med = (s.elapsed_time(e) / args.iters) if member else 0.0
        t = torch.tensor([med if member else 1e9], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        tmin = t.item()
        t[0] = med
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        tmax = t.item()                    # 非成员贡献 0, 不会赢 max
        if rank == 0:
            bw = bus_bytes / (med / 1e3) / 1e9  # GB/s bus 带宽口径
            print(f"[t23comm] {name:12s} {gname:7s} {med:9.3f} ms  "
                  f"[cross-rank {tmin:.3f}..{tmax:.3f}]  busBW {bw:7.2f} GB/s",
                  flush=True)
            rows.append(dict(op=name, group=gname, npar=gn,
                             payload_mb=round(nbytes_full / 1e6, 1),
                             med_ms=round(med, 3), min_ms=round(tmin, 3),
                             max_ms=round(tmax, 3), bus_bw_gbps=round(bw, 2)))

    bf16 = torch.bfloat16
    # ---- world16（跨节点, 全世界组）----
    full = torch.randn(ROWS, COLS, dtype=bf16, device="cuda")
    shard = ROWS * COLS // world
    outv = torch.empty(shard, dtype=bf16, device="cuda")
    bench("ar_shard62", "world16", g_world, world,
          lambda: dist.all_reduce(full, group=g_world), full.numel() * 2,
          2 * (world - 1) / world * full.numel() * 2)
    bench("rs_shard62", "world16", g_world, world,
          lambda: dist.reduce_scatter_tensor(outv, full, group=g_world),
          full.numel() * 2, (world - 1) / world * full.numel() * 2)
    bench("ag_shard62", "world16", g_world, world,
          lambda: dist.all_gather_into_tensor(full.view(-1), outv, group=g_world),
          shard * world * 2, 2 * (world - 1) / world * shard * world * 2)
    whale = torch.randn(WHALE_ELEMS, dtype=bf16, device="cuda")
    bench("ar_whale336", "world16", g_world, world,
          lambda: dist.all_reduce(whale, group=g_world), WHALE_ELEMS * 2,
          2 * (world - 1) / world * WHALE_ELEMS * 2)
    small = torch.randn(512 * 1024, dtype=bf16, device="cuda")  # 1MB 时延探针
    bench("ar_1mb", "world16", g_world, world,
          lambda: dist.all_reduce(small, group=g_world), small.numel() * 2,
          2 * (world - 1) / world * small.numel() * 2)
    del full, outv, whale, small

    # ---- intra8（节点内参考, 同 shard 载荷 62.9MB）----
    fi = torch.randn(ROWS, COLS, dtype=bf16, device="cuda")
    oi = torch.empty(ROWS * COLS // 8, dtype=bf16, device="cuda")
    bench("ar_shard62", "intra8", g_intra, 8,
          lambda: dist.all_reduce(fi, group=g_intra), fi.numel() * 2,
          2 * 7 / 8 * fi.numel() * 2)
    bench("rs_shard62", "intra8", g_intra, 8,
          lambda: dist.reduce_scatter_tensor(oi, fi, group=g_intra),
          fi.numel() * 2, 7 / 8 * fi.numel() * 2)
    del fi, oi

    # ---- dp2（跨节点对, TP8×DP2 车辆 DP 梯度面真实形状）----
    fp = torch.randn(WHALE_ELEMS, dtype=bf16, device="cuda")
    ops = torch.empty(WHALE_ELEMS // 2, dtype=bf16, device="cuda")
    bench("ar_whale336", "dp2", g_dp2, 2,
          lambda: dist.all_reduce(fp, group=g_dp2), WHALE_ELEMS * 2,
          WHALE_ELEMS * 2)
    bench("rs_whale336", "dp2", g_dp2, 2,
          lambda: dist.reduce_scatter_tensor(ops, fp, group=g_dp2),
          WHALE_ELEMS * 2, int(WHALE_ELEMS * 2 * 0.5))
    del fp, ops

    if rank == 0:
        new = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "ts", "rdma_arm", "rccl", "op", "group", "npar", "payload_mb",
                "med_ms", "min_ms", "max_ms", "bus_bw_gbps"])
            if new:
                w.writeheader()
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            for r in rows:
                r.update(ts=ts, rdma_arm=rdma_arm, rccl=nccl_ver)
                w.writerow(r)
        print(f"[t23comm] csv -> {args.out} ({len(rows)} rows)", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
