#!/bin/bash
# enable_rdma.sh -- shca 400G RDMA 用户态装配（SCNet vcjob 每作业一次，幂等）
# 语义与用户提示词/hunderd_billon_model 实测版逐条一致；source 本文件后调
# enable_rdma。资产只读引用 /root/private_data/second_hygon/rdma-shca（勿改动）。
#
# 开关:
#   RDMA_DISABLE=1  跳过安装与环境变量，保持 TCP(NET/Socket) 对照组
#   SHCA_HOME=<dir> 资产不在默认路径时指定
# 判据（唯一可信）: NCCL_DEBUG=INFO 日志出现 NET/IBext_v8 = RDMA 生效；
#   NET/Socket = 回退。ibv_devinfo 不列 shca 不代表失败（只作诊断）。
# 噪音（不要修）: open fd/lockfd: No such file；Failed to find ncclCollNetPlugin
enable_rdma() {
    if [[ "${RDMA_DISABLE:-0}" == "1" ]]; then
        log "RDMA: RDMA_DISABLE=1 → 保持 TCP(NET/Socket)（对照组）"
        return 0
    fi
    local shca_home="${SHCA_HOME:-/root/private_data/second_hygon/rdma-shca}"   # site-specific: set SHCA_HOME
    local libd=/usr/lib/x86_64-linux-gnu
    if [[ ! -e "${libd}/libshca-rdmav34.so" ]]; then
        local deb
        deb="$(ls "${shca_home}"/shca-tools_*.deb 2>/dev/null | head -1 || true)"
        if [[ -z "$deb" || ! -f "${shca_home}/topo_lib/built-in-508-topo-input-tj-default.xml" \
              || ! -d "${shca_home}/topo_lib/lib" ]]; then
            log "RDMA[warn]: shca 资产不完整(${shca_home})：缺 ${deb:-deb}/topo → 回退 TCP，不阻断"
            return 0
        fi
        log "RDMA: 安装 shca 用户态（离线 dpkg -x ~10s，幂等）: ${deb}"
        rm -rf /tmp/shca-install && mkdir -p /tmp/shca-install
        dpkg -x "$deb" /tmp/shca-install || { log "RDMA[warn]: dpkg -x 失败 → 回退 TCP"; return 0; }
        cp -a "/tmp/shca-install${libd}/." "${libd}/"
        # 遮蔽处理：镜像自带更高版本号 Mellanox 库(*47*)会盖住 shca 版(*44*)
        rm -f "${libd}/libibverbs.so.1.14.47.0" "${libd}/libmlx5.so.1.24.47.0" \
              "${libd}/librdmacm.so.1.3.47.0" "${libd}"/libibverbs.so.1.*47* 2>/dev/null || true
        ln -sf libibverbs.so.1.14.44.0 "${libd}/libibverbs.so.1"
        ln -sf librdmacm.so.1.3.44.0   "${libd}/librdmacm.so.1"
        ln -sf libmlx5.so.1.24.44.0    "${libd}/libmlx5.so.1"
        mkdir -p /etc/libibverbs.d /usr/etc/libibverbs.d /etc/cfgFile
        cp -a /tmp/shca-install/usr/etc/libibverbs.d/. /etc/libibverbs.d/  2>/dev/null || true
        cp -a /tmp/shca-install/usr/etc/libibverbs.d/. /usr/etc/libibverbs.d/ 2>/dev/null || true
        cp -a /tmp/shca-install/etc/cfgFile/. /etc/cfgFile/ 2>/dev/null || true
        mkdir -p /usr/local/bin && cp -a /tmp/shca-install/usr/bin/. /usr/local/bin/ 2>/dev/null || true
        ldconfig
        [[ -e "${libd}/libshca-rdmav34.so" ]] || { log "RDMA[warn]: 安装后仍无 libshca-rdmav34.so → 回退 TCP"; return 0; }
        log "RDMA: shca 用户态就位"
    else
        log "RDMA: shca 用户态已就位（跳过）"
    fi
    # 管理员配方：外置网络插件 + 平台拓扑文件（librccl-net-shca.so 在 topo_lib/lib）
    export NCCL_IB_DISABLE=0
    export NCCL_NET_PLUGIN=shca
    export NCCL_IB_HCA="${NCCL_IB_HCA:-shca_0:1,shca_1:1,shca_2:1,shca_3:1}"
    export NCCL_TOPO_FILE="${shca_home}/topo_lib/built-in-508-topo-input-tj-default.xml"
    export LD_LIBRARY_PATH="${shca_home}/topo_lib/lib:${LD_LIBRARY_PATH:-}"
    local hcas
    hcas="$(ibv_devinfo -l 2>/dev/null | grep '^shca_' | tr '\n' ' ' || true)"
    if [[ -n "$hcas" ]]; then
        log "RDMA: shca 设备可见:${hcas}（最终判据仍是 NCCL 日志 NET/IBext_v8）"
    else
        log "RDMA: ibv_devinfo 未列出 shca（参考诊断，不阻断；run-8 实测不列出但 IBext_v8 全通）"
    fi
    return 0
}
