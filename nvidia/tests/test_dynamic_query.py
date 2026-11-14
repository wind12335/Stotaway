import unittest
from planner.original_a800.dynamic_query import make_query,update_cache
from tests.test_predictor_registry import route
from unittest.mock import patch
from planner.original_a800.distributed_inference import prepare_collectively


class DynamicQueryTest(unittest.TestCase):
    def test_safe_point_update(self):
        r=route('dx','direct_exchange','ag');q=make_query(r,(1024,4096,4096),support='supported',evidence='adapter')
        cache={};self.assertEqual(update_cache(cache,'qkv',q,{'choice':'dx'}),'qkv:1024:4096:4096')
        with self.assertRaises(ValueError):update_cache(cache,'qkv',q,{'choice':'stock'})
        with self.assertRaises(ValueError):update_cache({},'fc2',q,{'choice':'dx'})
        unsupported=make_query(r,(1024,4096,4096),support='unknown',evidence='not checked')
        with self.assertRaises(ValueError):update_cache({},'qkv',unsupported,{'choice':'dx'})
        with self.assertRaises(ValueError):make_query(r,(1025,4096,4096),support='supported',evidence='adapter')

    def test_graph_capture_does_not_launch_collective(self):
        r=route('dx','direct_exchange','ag');q=make_query(r,(1024,4096,4096),support='supported',evidence='test')
        with patch('torch.distributed.is_initialized',return_value=True), \
             patch('torch.distributed.all_gather_object') as gather, \
             patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.is_current_stream_capturing',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'outside CUDA graph capture'):
                prepare_collectively(None,[q])
            gather.assert_not_called()


if __name__=='__main__':unittest.main()
