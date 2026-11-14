import unittest
from planner.original_a800.multi_route_selector import MultiRouteSelector
from planner.original_a800.orthogonal_predictor import Sample
from planner.original_a800.route_policy import GroupedSample
from tests.test_predictor_registry import route


class MultiRouteTest(unittest.TestCase):
    def test_grouped_training(self):
        r=route('dx','direct_exchange','ag')
        rows=[GroupedSample(str(i),Sample(r,(1024*(i+1),4096,4096),1,.8)) for i in range(5)]
        p=MultiRouteSelector(r).fit(rows)
        self.assertEqual(p.predict(r,(1024,4096,4096))['choice'],'dx')
        with self.assertRaises(ValueError):MultiRouteSelector(r).fit(rows[:3])


if __name__=='__main__':unittest.main()
