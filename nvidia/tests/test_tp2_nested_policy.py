import unittest
from planner.original_a800.tp2_nested_policy import select_policy


def row(family, gain):
    return {'model': family, 'op': 'rs', 'M': 1024, 'K': 1024, 'N': 1024,
            'G': 1.0, 'C': .2, 'B': 2.0, 'stock_b2b': 1.0, 'dx_b2b': 1.0 / gain}


class TestNestedPolicy(unittest.TestCase):
    def test_consistent_training_benefit_picks_dx(self):
        kind, threshold, info = select_policy([row(str(i), 1.2) for i in range(4)], 2.0)
        self.assertEqual(kind, 'dx')
        self.assertIsNone(threshold)
        self.assertEqual(info['training_families'], 4)

    def test_consistent_training_loss_picks_stock(self):
        kind, _, _ = select_policy([row(str(i), .8) for i in range(4)], 2.0)
        self.assertEqual(kind, 'stock')

    def test_insufficient_families_returns_stock(self):
        kind, _, info = select_policy([row('one', 1.2), row('two', 1.2)], 2.0)
        self.assertEqual(kind, 'stock')
        self.assertEqual(info['reason'], 'insufficient training families')


if __name__ == '__main__':
    unittest.main()
