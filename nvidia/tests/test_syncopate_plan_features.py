import unittest
from planner.original_a800.syncopate_plan_features import summarize


class FakeSchedule:
    num_waves=2
    def __init__(self):
        from types import SimpleNamespace
        self.block_infos=[[SimpleNamespace(signal=None)],[SimpleNamespace(signal=object())]]
    def gen_block_shape_lists(self):return [[1024,5120],[1024,5120]]
    def gen_signal_lists(self):return [-1,0]


class FeatureTest(unittest.TestCase):
    def test_local_and_remote_work(self):
        data=summarize({0:{'a':FakeSchedule()},1:{'a':FakeSchedule()}},buffer_name='a',k=5120,n_local=3072,dtype_bytes=2)
        self.assertEqual(data[0].signaled_bytes,1024*5120*2)
        self.assertEqual(data[1].gemm_flops,2*2048*3072*5120)
        self.assertEqual(data[0].dependency_waves,1)
        with self.assertRaises(ValueError):summarize({1:{'a':FakeSchedule()}},buffer_name='a',k=5120,n_local=3072,dtype_bytes=2)


if __name__=='__main__':unittest.main()
