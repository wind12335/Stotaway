"""vLLM eager sequence-parallel projection using one external backend + DX."""
import os
from .multi_backend8 import MultiBackend8
from .vllm_sp import LlamaSequenceParallel


class MultiBackendLlamaSequenceParallel(LlamaSequenceParallel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi = None
        self.multi_path = os.environ['VS_MULTI_PLAN8']
        self.multi_hits = {}

    def _dispatcher(self, group):
        if self.multi is None:
            self.multi = MultiBackend8(self.multi_path, 'llama70_infer', group)
        return self.multi

    def _col(self, layer, shard, rows, group, site):
        return self._dispatcher(group).run(site, 'ag', shard.detach().contiguous(), self._weight(layer))

    def _row(self, layer, value, rows, group, site):
        return self._dispatcher(group).run(site, 'rs', value.detach().contiguous(), self._weight(layer))

    def close(self):
        if self.multi is not None:
            self.multi_hits = {str(k): v for k, v in self.multi.hits.items()}
            self.multi.close()
        super().close()
