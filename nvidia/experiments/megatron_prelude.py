import atexit
import os
import runpy
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import triton

print(f"STOWAWAY_ENV rank={os.environ.get('RANK')} python={sys.executable} "
      f"torch={torch.__version__} triton={triton.__version__} triton_path={triton.__file__}", flush=True)

if os.environ.get("STOWAWAY_QWEN36_LOCAL_SPEC") == "1":
    # Diagnostic hybrid block with local linears; allows testing the existing
    # DX adapter.  Not the upstream TE baseline and not a HF weight converter.
    from megatron.core.models.gpt import experimental_attention_variant_module_specs as hybrid_specs
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.backends import LocalSpecProvider
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
    from megatron.core.transformer.spec_utils import ModuleSpec

    def local_backend(config):
        if config.transformer_impl != "local":
            raise ValueError("diagnostic hybrid spec requires local transformer implementation")
        return LocalSpecProvider()

    def local_gated(config, backend=None):
        backend = backend or local_backend(config)
        return ModuleSpec(module=GatedDeltaNet,
                          submodules=GatedDeltaNetSubmodules(
                              in_proj=backend.column_parallel_linear(),
                              out_norm=backend.layer_norm(rms_norm=True),
                              out_proj=backend.row_parallel_linear()),
                          metainfo={"fuse_input_layernorm": False})

    def local_attention(config, backend=None):
        spec = get_gpt_layer_local_spec(normalization=config.normalization,
                                        qk_layernorm=config.qk_layernorm,
                                        qk_l2_norm=config.qk_l2_norm).submodules.self_attention
        spec.metainfo = {"fuse_input_layernorm": False}
        return spec

    hybrid_specs._get_backend_spec_provider = local_backend
    hybrid_specs.get_gated_delta_net_module_spec = local_gated
    hybrid_specs._get_self_attention_module_spec = local_attention
    print("STOWAWAY: experimental local Qwen hybrid specification", flush=True)

if os.environ.get("STOWAWAY_ROUTE_CAPTURE_DIR"):
    from megatron.core.transformer.moe.moe_layer import MoELayer
    original_route = MoELayer.route
    route_dir = Path(os.environ["STOWAWAY_ROUTE_CAPTURE_DIR"])
    route_dir.mkdir(parents=True, exist_ok=True)
    route_count = [0]

    def captured_route(module, *args, **kwargs):
        probabilities, routing = original_route(module, *args, **kwargs)
        route_count[0] += 1
        path = route_dir / f"rank{dist.get_rank()}_route{route_count[0]:04d}.pt"
        torch.save({"routing": routing.detach().cpu().clone(),
                    "probabilities": probabilities.detach().cpu().clone()}, path)
        return probabilities, routing

    MoELayer.route = captured_route

if os.environ.get("STOWAWAY_MOE_GG") == "1":
    from stowaway_nvidia.megatron_moe import SequentialExpertAdapter
    moe_adapter = SequentialExpertAdapter()
    moe_adapter.install()

if os.environ.get("STOWAWAY_MOE_EP") == "1":
    if any(os.environ.get(name) == "1" for name in ("STOWAWAY_DX", "STOWAWAY_BRS", "STOWAWAY_OPT_BRS", "STOWAWAY_MOE_GG")):
        raise RuntimeError("MoE EP diagnostic must run separately from other NVSHMEM/adapter diagnostics")
    from stowaway_nvidia.megatron_moe_ep import MegatronEPAdapter
    moe_ep_adapter = MegatronEPAdapter()
    moe_ep_adapter.install()

if os.environ.get("STOWAWAY_TE_INSPECT") == "1":
    from megatron.training import training as te_inspect_training
    original_setup_model = te_inspect_training.setup_model_and_optimizer

    def inspect_te_setup(*args, **kwargs):
        result = original_setup_model(*args, **kwargs)
        model = result[0]
        for index, chunk in enumerate(model):
            for name, module in chunk.named_modules():
                ub_name = getattr(module, "ub_name", None)
                if ub_name in ("qkv", "proj", "fc1", "fc2"):
                    flags = {flag: getattr(module, flag, None) for flag in
                             ("ub_overlap_ag_fprop", "ub_overlap_rs_fprop", "ub_overlap_ag_dgrad",
                              "ub_overlap_rs_dgrad", "ub_bulk_dgrad", "ub_bulk_wgrad")}
                    print(f"STOWAWAY_TE_LAYER rank={os.environ.get('RANK')} chunk={index} "
                          f"name={name} ub={ub_name} flags={flags}", flush=True)
        return result

    te_inspect_training.setup_model_and_optimizer = inspect_te_setup

if os.environ.get("STOWAWAY_TE_LOCAL_NORM") == "1":
    import megatron.core.models.backends as megatron_backends
    import megatron.core.models.gpt.gpt_layer_specs as gpt_specs
    from megatron.core.extensions.transformer_engine import HAVE_TE, TENorm

    if not HAVE_TE:
        raise RuntimeError("STOWAWAY_TE_LOCAL_NORM requires an importable Transformer Engine")
    megatron_backends.LNImpl = TENorm
    def te_local_layer_norm(self, rms_norm=False, for_qk=False, has_residual=False):
        return TENorm

    megatron_backends.LocalSpecProvider.layer_norm = te_local_layer_norm
    gpt_specs.LNImpl = TENorm
    print("STOWAWAY: local linear layers with Transformer Engine TENorm", flush=True)

if os.environ.get("STOWAWAY_DIAGNOSTIC_TORCH_NORM") == "1":
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    original_norm = WrappedTorchNorm.__new__

    def diagnostic_norm(cls, config, *args, **kwargs):
        sequence_parallel = config.sequence_parallel
        config.sequence_parallel = False
        try:
            module = original_norm(cls, config, *args, **kwargs)
        finally:
            config.sequence_parallel = sequence_parallel
        for parameter in module.parameters():
            parameter.sequence_parallel = sequence_parallel
        return module

    WrappedTorchNorm.__new__ = diagnostic_norm
    print("WARNING: diagnostic TorchNorm SP adapter enabled; not a production baseline", flush=True)

if os.environ.get("STOWAWAY_OPT_BRS") == "1":
    if os.environ.get("STOWAWAY_BRS") == "1" or os.environ.get("STOWAWAY_DX") == "1":
        raise RuntimeError("optimizer BRS diagnostic must be tested separately")
    from megatron.training import training as opt_brs_training
    from stowaway_nvidia.megatron_optimizer_brs import OptimizerBucketBRS
    from stowaway_nvidia.runtime import initialize as initialize_opt_brs

    opt_brs_state = {"adapter": None}
    original_opt_brs_initialize = opt_brs_training.initialize_megatron

    def initialize_opt_brs_engine(*args, **kwargs):
        result = original_opt_brs_initialize(*args, **kwargs)
        from megatron.core import parallel_state
        group = parallel_state.get_data_parallel_group()
        if dist.get_world_size() != 2 or dist.get_process_group_ranks(group) != [0, 1]:
            raise RuntimeError("optimizer BRS diagnostic requires DP=world=2")
        adapter = OptimizerBucketBRS(initialize_opt_brs(),
                                     int(os.environ.get("STOWAWAY_BRS_MIN_BYTES", "0")),
                                     int(os.environ.get("STOWAWAY_BRS_SLOTS", "1")))
        adapter.install()
        opt_brs_state["adapter"] = adapter
        return result

    opt_brs_training.initialize_megatron = initialize_opt_brs_engine

if os.environ.get("STOWAWAY_BRS") == "1" and os.environ.get("STOWAWAY_DX") != "1":
    from megatron.training import training as brs_training
    from stowaway_nvidia.megatron_brs import TensorParallelBRS
    from stowaway_nvidia.runtime import initialize as initialize_brs

    brs_state = {"adapter": None}
    original_brs_initialize = brs_training.initialize_megatron

    def initialize_brs_engine(*args, **kwargs):
        result = original_brs_initialize(*args, **kwargs)
        from megatron.core import parallel_state
        group = parallel_state.get_tensor_model_parallel_group()
        if dist.get_world_size() != 2 or dist.get_process_group_ranks(group) != [0, 1]:
            raise RuntimeError("diagnostic BRS requires the world TP group")
        adapter = TensorParallelBRS(initialize_brs(),
                                    int(os.environ.get("STOWAWAY_BRS_MIN_BYTES", "0")),
                                    int(os.environ.get("STOWAWAY_BRS_SLOTS", "1")))
        adapter.install()
        brs_state["adapter"] = adapter
        return result

    brs_training.initialize_megatron = initialize_brs_engine

if os.environ.get("STOWAWAY_DX") == "1":
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    from megatron.training import training
    from stowaway_nvidia.autograd import column_parallel, row_parallel
    from stowaway_nvidia.dx import DirectExchange
    from stowaway_nvidia.runtime import initialize
    dx_world = int(os.environ.get("STOWAWAY_DX_WORLD", "2"))
    if dx_world not in (2, 8):
        raise ValueError("STOWAWAY_DX_WORLD must be 2 or 8")
    if dx_world == 8:
        from stowaway_nvidia.dx8 import DirectExchange8
        if os.environ.get("STOWAWAY_TP2_LOOKUP"):
            raise ValueError("TP8 DX does not support TP2 calibration")

    state = {"cuda": None, "contexts": {}, "brs": None, "row": 0, "column": 0,
             "row_skip": 0, "column_skip": 0}
    original_initialize = training.initialize_megatron
    original_row = RowParallelLinear.forward
    original_column = ColumnParallelLinear._forward_impl
    allowed_sites_env = os.environ.get("STOWAWAY_DX_SITES", "qkv,proj,fc1,fc2")
    allowed_sites = set(filter(None, allowed_sites_env.split(",")))
    valid_sites = {"qkv", "proj", "fc1", "fc2"}
    if not allowed_sites <= valid_sites:
        raise ValueError(f"unknown DX sites: {sorted(allowed_sites - valid_sites)}")
    # Upstream local layers accept tp_comm_buffer_name but do not retain it;
    # preserve that explicit constructor argument for per-site decisions.
    for layer_class in (ColumnParallelLinear, RowParallelLinear):
        original_constructor = layer_class.__init__
        def constructor(layer, *args, _original=original_constructor, **kwargs):
            site = kwargs.get("tp_comm_buffer_name")
            _original(layer, *args, **kwargs)
            layer._stowaway_site = site
        layer_class.__init__ = constructor
    print(f"STOWAWAY_DX_SITES rank={os.environ.get('RANK')} sites={sorted(allowed_sites)}", flush=True)
    planner_path = os.environ.get("STOWAWAY_TP2_LOOKUP")
    shape_plan_path = os.environ.get("STOWAWAY_DX_SHAPE_PLAN")
    if shape_plan_path and (planner_path or dx_world != 8):
        raise ValueError("shape plan requires TP8 and cannot be combined with TP2 lookup")
    planner_state = {"lookup": None, "shape_plan": None, "decisions": {}}
    column_max = int(os.environ.get("STOWAWAY_COLUMN_MAX_N", "8192"))
    column_min = int(os.environ.get("STOWAWAY_COLUMN_MIN_N", "0"))
    column_widths_env = os.environ.get("STOWAWAY_COLUMN_WIDTHS", "")
    column_widths = ({int(width) for width in column_widths_env.split(",")}
                     if column_widths_env else None)
    if column_min < 0 or column_max < column_min or (column_widths is not None and
                                                     any(width < 1 for width in column_widths)):
        raise ValueError("invalid diagnostic column width selection")

    def initialize_engine(*args, **kwargs):
        result = original_initialize(*args, **kwargs)
        from megatron.core import parallel_state
        group = parallel_state.get_tensor_model_parallel_group()
        if dist.get_world_size() != dx_world or group.size() != dx_world:
            raise RuntimeError("diagnostic DX adapter requires one world TP group")
        if planner_path:
            from planner.tp2_lookup import Lookup, runtime_signature
            interpolate = os.environ.get("STOWAWAY_TP2_INTERPOLATE") == "1"
            lookup = Lookup(planner_path, runtime_signature(torch, triton), interpolate=interpolate)
            configs = [None] * dist.get_world_size()
            dist.all_gather_object(configs, (lookup.digest, interpolate))
            if len(set(configs)) != 1:
                raise RuntimeError("TP2 lookup profile or interpolation setting differs across ranks")
            planner_state["lookup"] = lookup
            print(f"STOWAWAY_TP2_LOOKUP rank={dist.get_rank()} sha256={lookup.digest} "
                  f"mode={lookup.profile['classification']} interpolate={interpolate}", flush=True)
        if shape_plan_path:
            from stowaway_nvidia.shape_plan import ShapePlan
            # Gather load errors as well as digests before any rank raises: a
            # missing or mismatched plan on one rank must not strand its peers.
            plan, error = None, None
            try:
                plan = ShapePlan(shape_plan_path, tp=dx_world, dtype='bfloat16',
                                 gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                                 triton_version=triton.__version__, cuda_build=torch.version.cuda,
                                 schedule=os.environ.get('STOWAWAY_DX8_SCHEDULE', 'fixed'),
                                 batch_ack=os.environ.get('STOWAWAY_DX8_BATCH_ACK', '0') == '1',
                                 backward=os.environ.get('STOWAWAY_DX8_BACKWARD', '0') == '1',
                                 brs=os.environ.get('STOWAWAY_BRS', '0') == '1',
                                 brs_min_bytes=int(os.environ.get('STOWAWAY_BRS_MIN_BYTES', '0')),
                                 brs_slots=int(os.environ.get('STOWAWAY_BRS_SLOTS', '1')),
                                 cuda_device_max_connections=os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', 'unset'))
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            checks = [None] * dist.get_world_size()
            dist.all_gather_object(checks, (plan.digest if plan else None, error))
            if any(e for _, e in checks) or len({digest for digest, _ in checks}) != 1:
                raise RuntimeError(f'diagnostic shape plan invalid or inconsistent: {checks}')
            planner_state['shape_plan'] = plan
            print(f'STOWAWAY_DX_SHAPE_PLAN rank={dist.get_rank()} sha256={plan.digest}', flush=True)
        state["cuda"] = initialize(expected_world_size=dx_world)
        if os.environ.get("STOWAWAY_BRS") == "1":
            from stowaway_nvidia.megatron_brs import TensorParallelBRS
            state["brs"] = TensorParallelBRS(state["cuda"],
                                             int(os.environ.get("STOWAWAY_BRS_MIN_BYTES", "0")),
                                             int(os.environ.get("STOWAWAY_BRS_SLOTS", "1")), dx_world)
            state["brs"].install()
        return result

    def exchange_for(rows, inner, columns, dtype, purpose):
        key = (purpose, rows, inner, columns, dtype)
        if key not in state["contexts"]:
            exchange_type = DirectExchange8 if dx_world == 8 else DirectExchange
            state["contexts"][key] = exchange_type(state["cuda"], dist.get_rank(), dx_world,
                                                     rows, inner, columns, dtype)
        return state["contexts"][key]

    def check_tp_group(group):
        if group is None or dist.get_process_group_ranks(group) != list(range(dx_world)):
            raise RuntimeError("diagnostic adapter requires the world TP group")

    def use_dx(site, rows, inner, columns, dtype):
        shape_plan = planner_state['shape_plan']
        if shape_plan is not None:
            decision = shape_plan.choose(site, rows, inner, columns,
                                         'bfloat16' if dtype == torch.bfloat16 else str(dtype))
            item = (site, rows, inner, columns, decision)
            if item not in planner_state['decisions']:
                planner_state['decisions'][item] = 0
                print(f'STOWAWAY_DX_SHAPE_CHOICE rank={dist.get_rank()} site={site} '
                      f'M={rows} K={inner} N={columns} choice={decision}', flush=True)
            planner_state['decisions'][item] += 1
            return decision == 'dx'
        lookup = planner_state["lookup"]
        if lookup is None:
            return True
        decision = lookup.choose(site, rows, inner, columns) if dtype == torch.bfloat16 else "stock"
        item = (site, rows, inner, columns, decision)
        if item not in planner_state["decisions"]:
            planner_state["decisions"][item] = 0
            print(f"STOWAWAY_TP2_CHOICE rank={dist.get_rank()} site={site} "
                  f"M={rows} K={inner} N={columns} choice={decision}", flush=True)
        planner_state["decisions"][item] += 1
        return decision == "dx"

    def row_forward(layer, input_):
        eligible = (state["cuda"] is not None and layer.sequence_parallel and layer.input_is_parallel
                    and getattr(layer, "_stowaway_site", None) in allowed_sites
                    and not layer.explicit_expert_comm and not layer.gradient_accumulation_fusion
                    and layer.bias is None and input_.ndim == 3
                    and input_.dtype in (torch.float16, torch.bfloat16)
                    and input_.shape[0] % dx_world == 0 and input_.is_contiguous()
                    and layer.weight.requires_grad)
        if not eligible:
            state["row_skip"] += 1
            return original_row(layer, input_)
        rows = input_.shape[0] * input_.shape[1]
        check_tp_group(layer.tp_group)
        if not use_dx(layer._stowaway_site, rows, input_.shape[2], layer.weight.shape[0], input_.dtype):
            state["row_skip"] += 1
            return original_row(layer, input_)
        exchange = exchange_for(rows, input_.shape[2], layer.weight.shape[0], input_.dtype, "row")
        backward_exchange = (exchange_for(rows, layer.weight.shape[0], input_.shape[2], input_.dtype, "backward")
                             if dx_world == 2 or os.environ.get("STOWAWAY_DX8_BACKWARD") == "1" else None)
        output = row_parallel(input_.view(rows, -1), layer.weight, exchange, backward_exchange)
        state["row"] += 1
        return output.view(input_.shape[0] // dx_world, input_.shape[1], -1), None

    def column_forward(layer, input, weight, *args, **kwargs):
        eligible = (state["cuda"] is not None and not args and kwargs.get("sequence_parallel")
                    and getattr(layer, "_stowaway_site", None) in allowed_sites
                    and kwargs.get("bias") is None and not kwargs.get("gradient_accumulation_fusion")
                    and not kwargs.get("allreduce_dgrad") and kwargs.get("grad_output_buffer") is None
                    and input.ndim == 3 and input.is_contiguous() and weight.requires_grad
                    and column_min <= weight.shape[0] <= column_max
                    and (column_widths is None or weight.shape[0] in column_widths)
                    and input.dtype in (torch.float16, torch.bfloat16))
        if not eligible:
            state["column_skip"] += 1
            return original_column(layer, input, weight, *args, **kwargs)
        rows = input.shape[0] * input.shape[1] * dx_world
        check_tp_group(kwargs.get("tp_group"))
        if not use_dx(layer._stowaway_site, rows, input.shape[2], weight.shape[0], input.dtype):
            state["column_skip"] += 1
            return original_column(layer, input, weight, *args, **kwargs)
        exchange = exchange_for(rows, input.shape[2], weight.shape[0], input.dtype, "column")
        reduce_fn = None
        if state["brs"] is not None:
            from megatron.core.tensor_parallel import layers
            group = kwargs["tp_group"]
            def reduce_fn(out, value, async_op=False):
                return layers.dist_reduce_scatter_func(out, value, group=group, async_op=async_op)
        output = column_parallel(input.view(-1, input.shape[2]), weight, exchange, reduce_fn)
        state["column"] += 1
        return output.view(input.shape[0] * dx_world, input.shape[1], -1)

    training.initialize_megatron = initialize_engine
    RowParallelLinear.forward = row_forward
    ColumnParallelLinear._forward_impl = column_forward

    @atexit.register
    def report_counts():
        print(f"STOWAWAY_DIAGNOSTIC_COUNTS rank={os.environ.get('RANK')} "
              f"row={state['row']} column={state['column']} row_skip={state['row_skip']} "
              f"column_skip={state['column_skip']}", flush=True)

elif os.environ.get("STOWAWAY_TP2_LOOKUP") or os.environ.get("STOWAWAY_DX_SHAPE_PLAN"):
    raise RuntimeError("lookup or shape plan requires STOWAWAY_DX=1")

if os.environ.get("STOWAWAY_CAPTURE_DIR"):
    from megatron.training import training as capture_training

    original_train_step = capture_training.train_step
    capture_dir = Path(os.environ["STOWAWAY_CAPTURE_DIR"])
    capture_dir.mkdir(parents=True, exist_ok=True)
    capture_counter = [0]

    def captured_train_step(*args, **kwargs):
        result = original_train_step(*args, **kwargs)
        model = args[2] if len(args) > 2 else kwargs["model"]
        tensors = {}
        for index, chunk in enumerate(model):
            for name, parameter in chunk.named_parameters():
                tensors[f"{index}:{name}"] = parameter.detach().cpu().clone()
        capture_counter[0] += 1
        path = capture_dir / f"rank{dist.get_rank()}_step{capture_counter[0]}.pt"
        torch.save(tensors, path)
        print(f"STOWAWAY_DIAGNOSTIC_CAPTURE {path}", flush=True)
        return result

    capture_training.train_step = captured_train_step

if os.environ.get("STOWAWAY_GRAD_CAPTURE_DIR"):
    from megatron.training import training as grad_training

    original_grad_train_step = grad_training.train_step
    grad_capture_dir = Path(os.environ["STOWAWAY_GRAD_CAPTURE_DIR"])
    grad_capture_dir.mkdir(parents=True, exist_ok=True)
    grad_capture_counter = [0]

    def grad_captured_train_step(*args, **kwargs):
        positional = list(args)
        forward_backward = positional[6] if len(positional) > 6 else kwargs["forward_backward_func"]

        def capture_forward_backward(*forward_args, **forward_kwargs):
            result = forward_backward(*forward_args, **forward_kwargs)
            model = forward_kwargs.get("model")
            tensors = {}
            for index, chunk in enumerate(model):
                for name, parameter in chunk.named_parameters():
                    gradient = getattr(parameter, "main_grad", None)
                    if gradient is not None:
                        tensors[f"{index}:{name}"] = gradient.detach().cpu().clone()
            grad_capture_counter[0] += 1
            path = grad_capture_dir / f"rank{dist.get_rank()}_step{grad_capture_counter[0]}.pt"
            torch.save(tensors, path)
            print(f"STOWAWAY_DIAGNOSTIC_GRADS {path}", flush=True)
            return result

        if len(positional) > 6:
            positional[6] = capture_forward_backward
        else:
            kwargs["forward_backward_func"] = capture_forward_backward
        return original_grad_train_step(*positional, **kwargs)

    grad_training.train_step = grad_captured_train_step

if len(sys.argv) < 2:
    raise SystemExit("usage: megatron_prelude.py <pretrain_gpt.py> [arguments]")
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
if os.environ.get("STOWAWAY_MOE_EP") == "1":
    print(f"STOWAWAY_MOE_EP rank={dist.get_rank()} hits={moe_ep_adapter.hits} fallbacks={moe_ep_adapter.fallbacks} "
          f"pre_fc2_scale={moe_ep_adapter.pre_fc2_scale}", flush=True)
    moe_ep_adapter.close()
    if moe_ep_adapter.cuda is not None:
        from stowaway_nvidia.runtime import finalize
        finalize()
if os.environ.get("STOWAWAY_MOE_GG") == "1":
    print(f"STOWAWAY_MOE_GG rank={dist.get_rank()} hits={moe_adapter.hits} fallbacks={moe_adapter.fallbacks}", flush=True)
    moe_adapter.close()
if os.environ.get("STOWAWAY_OPT_BRS") == "1" and opt_brs_state["adapter"] is not None:
    from stowaway_nvidia.runtime import finalize

    adapter = opt_brs_state["adapter"]
    print(f"STOWAWAY_OPT_BRS_COUNTS rank={dist.get_rank()} calls={adapter.calls} "
          f"fallbacks={adapter.fallbacks} batches={adapter.batches} "
          f"multi_bucket_batches={adapter.multi_bucket_batches}", flush=True)
    adapter.close()
    finalize()
if os.environ.get("STOWAWAY_BRS") == "1" and os.environ.get("STOWAWAY_DX") != "1" and brs_state["adapter"] is not None:
    from stowaway_nvidia.runtime import finalize

    adapter = brs_state["adapter"]
    print(f"STOWAWAY_BRS_COUNTS rank={dist.get_rank()} hits={adapter.hits} "
          f"fallbacks={adapter.fallbacks}", flush=True)
    adapter.close()
    finalize()
if os.environ.get("STOWAWAY_DX") == "1" and state["cuda"] is not None:
    from stowaway_nvidia.runtime import finalize

    if state["brs"] is not None:
        print(f"STOWAWAY_BRS_COUNTS rank={dist.get_rank()} hits={state['brs'].hits} "
              f"fallbacks={state['brs'].fallbacks}", flush=True)
        state["brs"].close()
    for context in reversed(list(state["contexts"].values())):
        context.close()
    state["contexts"].clear()
    finalize()
