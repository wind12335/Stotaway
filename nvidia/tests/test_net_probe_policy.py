import unittest
from planner.original_a800.net_probe_policy import select_paired

class NetProbePolicyTest(unittest.TestCase):
    def test_cost_and_dispersion_affect_choice(self):
        pairs=[{'stock':1.,'dx':.9},{'stock':1.,'dx':.91},{'stock':1.,'dx':.89}]
        self.assertEqual(select_paired(pairs,probe_ms=15,remaining_calls=100)['choice'],'stock')
        self.assertEqual(select_paired(pairs,probe_ms=1,remaining_calls=100)['choice'],'dx')
        noisy=[{'stock':1.,'dx':.8},{'stock':1.,'dx':1.1}]
        self.assertEqual(select_paired(noisy,probe_ms=0,remaining_calls=100)['choice'],'stock')
        with self.assertRaises(ValueError):select_paired(pairs,probe_ms=float('nan'),remaining_calls=10)
        with self.assertRaises(ValueError):select_paired(pairs[:1],probe_ms=0,remaining_calls=10)
