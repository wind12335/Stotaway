import unittest
from planner.original_a800.realized_net import summarize

class RealizedNetTest(unittest.TestCase):
    def test_probe_cost_is_included_before_ratio(self):
        rows=[{'id':'a','stock_ms':100.,'optimized_ms':95.},{'id':'b','stock_ms':100.,'optimized_ms':95.}]
        self.assertGreater(summarize(rows)['geomean_net_speedup'],1.)
        result=summarize(rows,incremental_setup_ms=10.,completed_units_per_block=100)
        self.assertLess(result['geomean_net_speedup'],1.)
        self.assertEqual(result['slower_blocks'],2)
        self.assertAlmostEqual(result['median_net_saving_ms_per_unit'],-.05)
        with self.assertRaises(ValueError):summarize(rows+rows)
