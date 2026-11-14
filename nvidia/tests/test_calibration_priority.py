import unittest
from planner.original_a800.calibration_priority import Candidate,rank_probes


class PriorityTest(unittest.TestCase):
    def test_cost_and_uncertainty(self):
        known=Candidate('known',-.1,0,10,1)
        unknown=Candidate('unknown',0,.1,10,1)
        out=rank_probes([known,unknown],remaining_steps=100)
        self.assertEqual(out[0]['name'],'unknown')
        self.assertEqual(out[1]['optimistic_net_ms'],-1)
        self.assertAlmostEqual(rank_probes([unknown],remaining_steps=1)[0]['expected_perfect_information_ms_per_call'],.1/(2*3.141592653589793)**.5)
        with self.assertRaises(ValueError):rank_probes([unknown],remaining_steps=0)


if __name__=='__main__':unittest.main()
