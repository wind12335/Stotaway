import json
import tempfile
import unittest
from pathlib import Path
from planner.original_a800.reduced_probe_policy import ReducedProbePolicy
from planner.original_a800.reduced_probe_runtime import prepare
from planner.tp2_lookup import Lookup


class RuntimeTest(unittest.TestCase):
    def test_bundle_inventory_and_forward_phase(self):
        runtime={'tp':2,'dtype':'bfloat16','gpu':'A800','torch':'2','triton':'3','cuda_build':'12',
                 'cuda_device_max_connections':'1'}
        sig=['dx',2,'bfloat16','A800','2','3','12','1',16,'','',3]
        rows=[{'family':str(i),'op':'rs','M':1024*(i+1),'N':4096,'K':2048,'full_ms':.2*(i+1),
               'split_ms':.22*(i+1),'comm_ms':.04*(i+1),'stock_ms':.24*(i+1),'overlap_ms':.2*(i+1)} for i in range(8)]
        policy=ReducedProbePolicy().fit(rows,op='rs').to_dict()
        model={'schema':'stowaway-fragmentation-freeze-v1','signature':sig,'models':{'rs':policy},
               'targets':[{'model':'x','site':'o_proj','op':'rs','M':1024,'N':4096,'K':2048}]}
        components={'metadata':{'tp':2,'gpu':'A800','torch':'2','triton':'3','cuda_build':'12',
                    'cuda_device_max_connections':'1','batch':16,'warmup':3},
                    'results':[{'model':'x','site':'o_proj','op':'rs','M':1024,'N':4096,'K':2048,
                    'measurements':{'full':{'median_ms':.2},'split':{'median_ms':.22}}}]}
        workload={'seq_length':1024,'micro_batch_size':1};cap={'proj:1024:2048:4096':{'support':'supported','evidence':'verified'}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m=root/'model.json';c=root/'components.json';p=root/'plan.json'
            m.write_text(json.dumps(model));c.write_text(json.dumps(components))
            kwargs={'runtime':runtime,'workload':workload,'capabilities':cap,'phase':'forward','target_model':'x'}
            out=prepare(m,[c],**kwargs)
            p.write_text(json.dumps(out));lookup=Lookup(p,runtime)
            self.assertEqual(lookup.choose('proj',1024,2048,4096),'dx')
            self.assertEqual(lookup.choose('proj',2048,2048,4096),'stock')
            with self.assertRaises(ValueError):prepare(m,[c],**{**kwargs,'phase':'training'})
            with self.assertRaises(ValueError):prepare(m,[c],**{**kwargs,'capabilities':{}})
            with self.assertRaises(ValueError):prepare(m,[c],**{**kwargs,'target_model':'missing'})
            with self.assertRaises(ValueError):prepare(m,[c],**{**kwargs,'workload':{'seq_length':-1024,'micro_batch_size':-1}})
            cap['proj:1024:2048:4096']['support']='unknown'
            self.assertEqual(prepare(m,[c],**kwargs)['entries']['proj:1024:2048:4096']['choice'],'stock')
            model['targets'].append({**model['targets'][0],'M':4096})
            m.write_text(json.dumps(model))
            self.assertEqual(len(prepare(m,[c],**kwargs)['entries']),1)
            components['results'][0]['op']='ag';c.write_text(json.dumps(components))
            with self.assertRaisesRegex(ValueError,'operation mismatch'):prepare(m,[c],**kwargs)

    def test_local_residual_bundle(self):
        from planner.original_a800.local_residual_policy import LocalResidualPolicy
        runtime={'tp':2,'dtype':'bfloat16','gpu':'A800','torch':'2','triton':'3','cuda_build':'12',
                 'cuda_device_max_connections':'1'}
        sig=['dx',2,'bfloat16','A800','2','3','12','1',16,'','',3]
        rows=[{'family':str(i),'op':'rs','M':1024*(i+1),'N':4096,'K':2048,'full_ms':.2*(i+1),
               'split_ms':.22*(i+1),'comm_ms':.04*(i+1),'stock_ms':.24*(i+1),'overlap_ms':.2*(i+1)} for i in range(8)]
        model={'schema':'stowaway-fragmentation-freeze-v1','signature':sig,
               'models':{'rs':LocalResidualPolicy().fit(rows,op='rs').to_dict()},
               'targets':[{'model':'x','site':'o_proj','op':'rs','M':1024,'N':4096,'K':2048}]}
        components={'metadata':{'tp':2,'gpu':'A800','torch':'2','triton':'3','cuda_build':'12',
                    'cuda_device_max_connections':'1','batch':16,'warmup':3},
                    'results':[{'model':'x','site':'o_proj','op':'rs','M':1024,'N':4096,'K':2048,
                    'measurements':{'full':{'median_ms':.2},'split':{'median_ms':.22}}}]}
        with tempfile.TemporaryDirectory() as directory:
            m=Path(directory)/'model.json';c=Path(directory)/'components.json'
            m.write_text(json.dumps(model));c.write_text(json.dumps(components))
            out=prepare(m,[c],runtime=runtime,workload={'seq_length':1024,'micro_batch_size':1},
                        capabilities={'proj:1024:2048:4096':{'support':'supported','evidence':'verified'}},
                        phase='forward',target_model='x')
            self.assertEqual(out['predictor_model_family'],'stowaway-local-residual-v1')
            self.assertEqual(out['entries']['proj:1024:2048:4096']['choice'],'dx')


if __name__=='__main__':unittest.main()
