import unittest
from planner.original_a800.pipeline_subset import Operation, pipeline_makespan, predict_combinations, fit_linear_residual


class TestPipelineSubset(unittest.TestCase):
    def test_pipeline_dependency_and_tail(self):
        self.assertEqual(pipeline_makespan([2, 2], [3, 3]), 8)
        self.assertEqual(pipeline_makespan([1, 1, 1], [0, 0, 0]), 3)
        with self.assertRaises(ValueError):
            pipeline_makespan([1], [1, 2])

    def test_phase_counts_and_interaction_change_choice(self):
        ops = [Operation('qkv', 'forward', 2, 1, 2),
               Operation('proj', 'forward', 2, 1, 2),
               Operation('proj', 'backward', 2, 3, 1)]
        result = predict_combinations('dx', ops, {('qkv', 'proj'): 3})
        self.assertEqual(result['best']['sites'], ['qkv'])
        self.assertEqual(result['best']['delta_ms'], -2)

    def test_unsupported_site_and_guard_are_never_selected(self):
        ops = [Operation('qkv', 'forward', 2, 1, 1, False),
               Operation('proj', 'forward', 2, 1, 1, True)]
        result = predict_combinations('flux', ops, pessimism_ms={'proj': 1.1})
        self.assertEqual(result['best']['sites'], [])
        self.assertEqual(len(result['candidates']), 2)

    def test_reject_invalid_input(self):
        with self.assertRaises(ValueError):
            predict_combinations('stock', [Operation('x', 'f', 1, 1, 1)])
        with self.assertRaises(ValueError):
            predict_combinations('dx', [Operation('x', 'f', 1, 1, 0)])

    def test_nonnegative_calibration(self):
        p = fit_linear_residual([1, 2, 3], [3, 5, 7])
        self.assertAlmostEqual(p['scale'], 2)
        self.assertAlmostEqual(p['fixed_ms'], 1)
        with self.assertRaises(ValueError):
            fit_linear_residual([1, 1], [2, 3])


if __name__ == '__main__':
    unittest.main()
