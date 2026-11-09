"""Hygon DCU (gfx936) kernels. Import modules directly (no eager imports: each module JIT-compiles
triton kernels and some need pydushmem initialised first, see common_ops.init_dcu)."""
