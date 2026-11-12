import unittest

from planner.original_a800.combination_planner import CombinationPlanner, Observation


class CombinationPlannerTest(unittest.TestCase):
    def test_unobserved_single_site_is_not_inferred_from_a_group(self):
        planner = CombinationPlanner('dx', ('proj', 'fc2'), {'proj': -1, 'fc2': -2}, 'batch')
        planner.fit([Observation('batch', (1, 1), -4)])
        self.assertIsNone(planner.estimate((1, 0)))
        self.assertIsNone(planner.estimate((0, 1)))
        self.assertAlmostEqual(planner.estimate((1, 1)), -4)
        self.assertEqual(planner.select()['choices'], {'proj': 'dx', 'fc2': 'dx'})
        self.assertEqual(planner.select()['estimable_options'], 2)

    def test_measured_interaction_can_reverse_an_additive_choice(self):
        p = CombinationPlanner('flux', ('proj', 'fc2'), {'proj': -2, 'fc2': -2},
                               'same-batch', interaction_order=2)
        p.fit([Observation('same-batch', (1, 0), -2),
               Observation('same-batch', (0, 1), -1),
               Observation('same-batch', (1, 1), 1)])
        self.assertEqual(p.select()['choices'], {'proj': 'flux', 'fc2': 'stock'})
        self.assertAlmostEqual(p.estimate((1, 1)), 1)

    def test_reject_foreign_or_missing_calibration(self):
        p = CombinationPlanner('dx', ('qkv',), {'qkv': -1}, 'batch-a')
        with self.assertRaises(RuntimeError):
            p.select()
        with self.assertRaises(ValueError):
            p.fit([Observation('batch-b', (1,), -3)])
        with self.assertRaises(ValueError):
            p.fit([Observation('batch-a', (0,), 1)])

    def test_unsupported_sites_and_margin(self):
        p = CombinationPlanner('triton-dist', ('qkv', 'fc2'), {'qkv': -1, 'fc2': -1},
                               'batch', min_gain_ms=2)
        p.fit([Observation('batch', (1, 0), -1), Observation('batch', (0, 1), -1)])
        self.assertEqual(p.select(allowed={'fc2'})['choices'], {'qkv': 'stock', 'fc2': 'stock'})
        with self.assertRaises(ValueError):
            p.select(allowed={'unavailable'})

    def test_unmeasured_combination_requires_explicit_extrapolation(self):
        p = CombinationPlanner('dx', ('qkv', 'fc2'), {'qkv': -1, 'fc2': -1}, 'batch')
        p.fit([Observation('batch', (1, 0), -1), Observation('batch', (0, 1), -1)])
        self.assertEqual(p.select()['estimable_options'], 3)
        self.assertEqual(p.select(allow_model_extrapolation=True)['estimable_options'], 4)
        self.assertEqual(p.select(allow_model_extrapolation=True)['choices'], {'qkv': 'dx', 'fc2': 'dx'})

    def test_measured_combinations_use_measured_deltas_even_if_model_misspecified(self):
        p = CombinationPlanner('dx', ('a', 'b'), {'a': -1, 'b': -1}, 'batch')
        p.fit([Observation('batch', (1, 0), -1), Observation('batch', (0, 1), -1),
               Observation('batch', (1, 1), 4)])
        self.assertNotEqual(p.select()['choices'], {'a': 'dx', 'b': 'dx'})


if __name__ == '__main__':
    unittest.main()
