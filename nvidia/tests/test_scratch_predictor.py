import json
import tempfile
import unittest
from pathlib import Path
from planner.original_a800.scratch_predictor import freeze,evaluate
from planner.original_a800.validate_tp2_new import load


def metadata(m,ops):
    d={'tp':2,'gpu':'A800','torch':'2','triton':'3','cuda_build':'12',
       'cuda_device_max_connections':'1','batch':16,'warmup':3,'rs_actual_path':'direct_exchange_rs_scratch'}
    if ops:d.update(dtype='bfloat16',checks=2,rs_scratch=True)
    return d


def row(i,m,ops):
    obj={'model':f'v{i}','site':'fc2','op':'rs','M':m,'N':4096+i*128,'K':2048+i*128,
         'elapsed_wall_ms':5.}
    obj['measurements']=({'stock':{'median_ms':.8+i*.01},'dx':{'median_ms':.7+i*.01},
                          'comm':{'median_ms':.2+i*.005}} if ops else
                         {'full':{'median_ms':.6+i*.01},'split':{'median_ms':.65+i*.01}})
    return obj


class ScratchPredictorTest(unittest.TestCase):
    def test_isolation_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);ops=p/'ops.json';stages=p/'stages.json';target=p/'target.json';truth=p/'truth.json';frozen=p/'frozen.json'
            ops.write_text(json.dumps({'metadata':metadata(1024,True),'results':[row(i,1024,True) for i in range(8)]}))
            stages.write_text(json.dumps({'metadata':metadata(1024,False),'results':[row(i,1024,False) for i in range(8)]}))
            target.write_text(json.dumps({'metadata':metadata(1024,False),'results':[row(9,1024,False)]}))
            truth.write_text(json.dumps({'metadata':metadata(1024,True),'results':[row(9,1024,True)]}))
            data=freeze([ops],[stages],[target]);frozen.write_text(json.dumps(data))
            self.assertEqual(evaluate(frozen,[truth])['summary']['model']['count'],1)
            target.write_text(json.dumps({'metadata':metadata(1024,False),'results':[row(0,1024,False)]}))
            with self.assertRaisesRegex(ValueError,'family in training'):freeze([ops],[stages],[target])
            with self.assertRaisesRegex(ValueError,'original DX models'):load([ops])
            wrong=json.loads(truth.read_text());wrong['metadata']['rs_actual_path']='direct_exchange';truth.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError,'scratch TP2'):evaluate(frozen,[truth])


if __name__=='__main__':unittest.main()
