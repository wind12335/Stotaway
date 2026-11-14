import json
import tempfile
import unittest
from pathlib import Path

from planner.original_a800.export_paircal_mask import RUNTIME, WORKLOAD
from planner.original_a800.tp2_online_bundle import OnlineBundle, build_bundle


class OnlineBundleTest(unittest.TestCase):
    def test_isolation_runtime_and_unknown_fallback(self):
        def source(m):
            metadata={**RUNTIME,'M':m,'batch':16,'checks':1,'warmup':3,
                      'classification':'diagnostic unfused site forward+backward'}
            rows=[{'site':'full_proj','M':m,'N':5120,'K':3072,
                   'measurements':{'stock':{'median_ms':1.},'dx':{'median_ms':.8}}}]
            return {'metadata':metadata,'results':rows}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);train=[]
            for m in (1024,4096):
                p=root/f'{m}.json';p.write_text(json.dumps(source(m)));train.append(p)
            cap=root/'2048.json';cap.write_text(json.dumps(source(2048)))
            data=build_bundle(train,cap,RUNTIME,WORKLOAD);path=root/'bundle.json';path.write_text(json.dumps(data))
            self.assertEqual(data['models']['proj']['route']['topology'],'unverified')
            mismatched=source(1024);mismatched['metadata']['topology']='nv8'
            mismatch=root/'mismatch.json';mismatch.write_text(json.dumps(mismatched))
            with self.assertRaisesRegex(ValueError,'topology metadata mismatch'):
                build_bundle([mismatch],cap,RUNTIME,WORKLOAD)
            bundle=OnlineBundle(path,RUNTIME,WORKLOAD)
            q=bundle.queries[0];prediction=bundle.selector.predict(q.route,q.shape,capability=q.capability)
            self.assertEqual(bundle.choices([prediction])['proj:2048:3072:5120'],'dx')
            self.assertNotIn('proj:3072:3072:5120',bundle.choices([prediction]))
            self.assertEqual(bundle.choose(bundle.choices([prediction]),'proj',3072,3072,5120),'stock')
            self.assertEqual(bundle.choose(bundle.choices([prediction]),'proj',2048,3072,5120,dtype='float16'),'stock')
            data['overrides']={'proj:2048:3072:5120':{'choice':'stock','source_sha256':'a'*64}}
            path.write_text(json.dumps(data));overridden=OnlineBundle(path,RUNTIME,WORKLOAD)
            self.assertEqual(overridden.choices([prediction])['proj:2048:3072:5120'],'stock')
            with self.assertRaises(ValueError):OnlineBundle(path,RUNTIME,{**WORKLOAD,'seq_length':1024})
            with self.assertRaises(ValueError):build_bundle(train+[cap],cap,RUNTIME,WORKLOAD)
            data['queries'][0]['support']='unknown';path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):OnlineBundle(path,RUNTIME,WORKLOAD)


if __name__=='__main__':unittest.main()
