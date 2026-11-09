"""lightop stub for DCU (das vLLM fork imports `lightop` at module scope
in mla/indexer.py and quantization/awq.py; the real wheel is absent).

All attributes resolve to _Stub objects that raise NotImplementedError
when CALLED (or when subscripted etc.), while nested attribute access
yields more stubs -- so `from lightop import op; op.foo(...)` imports
cleanly for model families we never run, and any genuinely needed call
fails loudly. bf16 llama serving touches none of them
(reshape_and_cache_cuda falls back to _custom_ops unless
VLLM_USE_OPT_RESHAPE_AND_CACHE=1). Replace with the real wheel when
available.
"""


class _Stub:
    def __init__(self, name):
        self._name = name

    def _fail(self, *a, **k):
        raise NotImplementedError(
            f"lightop stub: '{self._name}' called but the real lightop "
            "wheel is not installed (DCU stub).")

    __call__ = _fail

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Stub(f"{self._name}.{attr}")


def _reshape_and_cache_cuda(*a, **k):
    # das fork auto-enables VLLM_USE_OPT_RESHAPE_AND_CACHE for Qwen3, which
    # calls lightop.reshape_and_cache_cuda with the exact fa_utils signature
    # -- delegate to vllm's own compiled op (lazy import: _custom_ops
    # imports lightop at module top, so only resolve at call time).
    from vllm._custom_ops import reshape_and_cache_cuda
    return reshape_and_cache_cuda(*a, **k)


_DELEGATES = {
    "reshape_and_cache_cuda": _reshape_and_cache_cuda,
}


def __getattr__(name):
    if name in _DELEGATES:
        return _DELEGATES[name]
    return _Stub(f"lightop.{name}")
