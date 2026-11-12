import unittest
from dataclasses import replace

from planner.original_a800.backend_family import (
    Environment, Signature, MeasurementKey, OperatorEstimate, Occurrence,
    StepCalibration, plan,
)


class BackendFamilyTest(unittest.TestCase):
    def setUp(self):
        self.env = Environment('dx', 'hybrid', 'sm80', 'a800', 'nvswitch', 8,
                               'bf16', (('cuda', '12.4'), ('triton', '3.1')))
        ag = Signature(self.env, 'vendor_gemm', 'nvshmem', 'ag', (('chunks', '8'),), 'actual_dx')
        rs = replace(ag, direction='rs')
        self.q = MeasurementKey(ag, 'forward', (4096, 8192, 4096), 'b2b16')
        self.p = MeasurementKey(rs, 'backward', (4096, 4096, 8192), 'b2b16')
        self.ops = [Occurrence('qkv', self.q, 2), Occurrence('proj', self.p, 1)]
        self.est = [OperatorEstimate(self.q, 2, 1, .1, 'model-a'),
                    OperatorEstimate(self.p, 2, 1, .1, 'model-b')]

    def calibration(self, pairs):
        return StepCalibration(self.env, 'qwen-step',
                               (self.q.signature, self.p.signature), .1, pairs, 'same-context measured')

    def test_missing_step_returns_stock_and_counts_unknown_masks(self):
        out = plan(self.env, 'qwen-step', self.ops, self.est, None)
        self.assertEqual(out['best']['sites'], [])
        self.assertEqual(out['uncalibrated_masks'], 3)

    def test_unmeasured_pair_is_not_assumed_zero(self):
        out = plan(self.env, 'qwen-step', self.ops, self.est, self.calibration({}))
        self.assertEqual(out['best']['sites'], ['qkv'])
        self.assertEqual(out['uncalibrated_masks'], 1)
        out = plan(self.env, 'qwen-step', self.ops, self.est,
                   self.calibration({('proj', 'qkv'): 0.0}))
        self.assertEqual(out['best']['sites'], ['proj', 'qkv'])
        out = plan(self.env, 'qwen-step', self.ops, self.est,
                   self.calibration({('proj', 'qkv'): 2.0}))
        self.assertEqual(out['best']['sites'], ['qkv'])

    def test_context_signature_and_timing_isolation(self):
        for env in (replace(self.env, world_size=4), replace(self.env, dtype='float16'),
                    replace(self.env, method='flux', family='cuda_cutlass')):
            with self.assertRaises(ValueError):
                plan(env, 'qwen-step', self.ops, self.est, None)
        with self.assertRaises(ValueError):
            plan(self.env, 'different-workload', self.ops, self.est, self.calibration({}))
        with self.assertRaises(ValueError):
            plan(self.env, 'qwen-step', self.ops, self.est,
                 replace(self.calibration({}), signatures=(self.q.signature,)))
        single = replace(self.q, timing_mode='single')
        estimate = replace(self.est[0], key=single)
        out = plan(self.env, 'qwen-step', self.ops, [estimate, self.est[1]], self.calibration({}))
        self.assertEqual(out['unavailable_sites'], ['qkv'])
        self.assertEqual(out['best']['sites'], ['proj'])

    def test_phase_occurrences_and_uncertainty_gate(self):
        # A site is disabled if any occurrence is unavailable, including backward.
        ops = self.ops + [Occurrence('qkv', replace(self.q, phase='backward'), 1)]
        out = plan(self.env, 'qwen-step', ops, self.est, self.calibration({}))
        self.assertEqual(out['unavailable_sites'], ['qkv'])
        conservative = [replace(e, uncertainty_ms=2) for e in self.est]
        out = plan(self.env, 'qwen-step', self.ops, conservative, self.calibration({}))
        self.assertEqual(out['best']['sites'], [])

    def test_invariants(self):
        with self.assertRaises(ValueError):
            replace(self.env, versions=(('cuda', '12'), ('cuda', '11')))
        with self.assertRaises(ValueError):
            replace(self.env, world_size=True)
        with self.assertRaises(ValueError):
            plan(self.env, 'qwen-step', self.ops, self.est + [self.est[0]], None)
        with self.assertRaises(ValueError):
            self.calibration({('qkv', 'proj'): 0.0})


if __name__ == '__main__':
    unittest.main()
