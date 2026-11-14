import unittest
from planner.original_a800.latency_guard import LatencyGuard

class LatencyGuardTest(unittest.TestCase):
    def test_switch_requires_two_consecutive_observations(self):
        g=LatencyGuard(1.,margin=.02,consecutive=2)
        self.assertEqual(g.observe('dx',1.03),'dx')
        self.assertEqual(g.observe('dx',.9),'dx')
        self.assertEqual(g.observe('dx',1.03),'dx')
        self.assertEqual(g.observe('dx',1.04),'stock')
        self.assertEqual(g.observe('stock',2.),'stock')
        with self.assertRaises(ValueError):g.observe('dx',.9)
        with self.assertRaises(ValueError):LatencyGuard(float('nan'))
