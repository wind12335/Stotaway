import json
import unittest

from planner.original_a800.guarded_hybrid_route import GuardedHybridRoute
from planner.original_a800.orthogonal_predictor import ContextRequirement
from planner.original_a800.predictor_registry import PredictorRegistry
from tests.test_predictor_registry import route


class GuardedHybridTest(unittest.TestCase):
    def test_guard_exact_fallback_and_roundtrip(self):
        r=route('flashoverlap','cutlass_nccl_wave','rs');rows=[]
        for i in range(5):
            rows.append((str(i),(1024,4096+i*128,4096),[1.]*8,[.8]*8))
        rows[0]=('0',rows[0][1],[1.]*8,[.99]*8)
        model=GuardedHybridRoute(r).fit(rows);shape=rows[0][1]
        self.assertEqual(model.choose(r,shape)['choice'],'stock')
        self.assertEqual(model.choose(r,rows[1][1])['choice'],'flashoverlap')
        self.assertEqual(model.choose(r,(2048,4096,4096))['source'],'route_local_fallback')
        restored=GuardedHybridRoute.from_dict(json.loads(json.dumps(model.to_dict())))
        self.assertEqual(restored.choose(r,shape),model.choose(r,shape))
        reg=PredictorRegistry();reg.register(model)
        self.assertEqual(reg.predict(r,shape,capability=ContextRequirement(r,shape,'supported','checked'))['choice'],'stock')
        bad=model.to_dict();bad['exact'][0]['choice']='flashoverlap'
        with self.assertRaises(ValueError):GuardedHybridRoute.from_dict(bad)


if __name__=='__main__':unittest.main()
