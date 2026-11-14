import unittest

from planner.original_a800.evaluate_nested_reduced_margin import select_margin


class MarginSelectionTest(unittest.TestCase):
    def test_penalizes_slow_decisions_without_losing_large_benefits(self):
        # 0.02 drops the small false-positive, preserving the profitable point.
        self.assertEqual(select_margin([(.01, -.04), (.10, .20)]), .02)
        # When the small predicted gain is actually beneficial, no margin wins.
        self.assertEqual(select_margin([(.01, .20), (.10, .20)]), 0.)


if __name__ == '__main__':
    unittest.main()
