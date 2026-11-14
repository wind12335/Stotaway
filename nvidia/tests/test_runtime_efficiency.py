import unittest
from planner.original_a800.runtime_efficiency import select_execution_mode


class RuntimeEfficiencyTest(unittest.TestCase):
    def test_savings_must_amortize_measured_calibration(self):
        blocks=[{'id':'a','stock_ms':100.,'optimized_ms':98.},
                {'id':'b','stock_ms':100.,'optimized_ms':99.}]
        self.assertEqual(select_execution_mode(blocks,calibration_ms=50,horizon=10)['choice'],'stock')
        self.assertEqual(select_execution_mode(blocks,calibration_ms=50,horizon=100)['choice'],'optimized')
        with self.assertRaises(ValueError):select_execution_mode(blocks[:1],calibration_ms=0,horizon=100)
        with self.assertRaises(ValueError):select_execution_mode(blocks+blocks,calibration_ms=0,horizon=100)


if __name__=='__main__':unittest.main()
