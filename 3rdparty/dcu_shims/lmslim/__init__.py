"""lmslim stub for DCU (das vLLM fork imports it unconditionally at module
scope; the real wheel is not on PyPI and absent on this box).

Every attribute resolves to a stub callable that raises NotImplementedError
WHEN CALLED, so unquantized (bf16) serving imports cleanly while any real
quantized path fails loudly instead of silently. Replace this package with
the real lmslim wheel when available.
"""


def _stub(name):
    def _call(*_a, **_k):
        raise NotImplementedError(
            f"lmslim stub: '{name}' called but the real lmslim wheel is "
            "not installed (DCU stub). Install the das lmslim wheel for "
            "quantized-model support.")
    return _call


def __getattr__(name):
    if name in ("quant_ops", "quant_tools"):
        import importlib
        return importlib.import_module(f"lmslim.quantize.quant_ops")
    return _stub(f"lmslim.{name}")
