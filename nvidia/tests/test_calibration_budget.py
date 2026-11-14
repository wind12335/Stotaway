import unittest
from planner.original_a800.calibration_budget import Amortization


class BudgetTest(unittest.TestCase):
    def test_breakeven_and_negative_gain(self):
        x=Amortization(1000,2)
        self.assertEqual(x.evaluate(500)['net_saving_ms'],0)
        self.assertEqual(x.evaluate(1)['breakeven_steps'],500)
        self.assertIsNone(Amortization(10,-1).evaluate(100)['breakeven_steps'])
        with self.assertRaises(ValueError):Amortization(-1,1)


if __name__=='__main__':unittest.main()
