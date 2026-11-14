import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from planner.original_a800.export_tp2_step_plan import export_step_plan, write_json_new
from planner.original_a800.orthogonal_predictor import ContextRequirement, RouteRegressor, Sample
from planner.original_a800.predictor_registry import PredictorRegistry, bind
from planner.original_a800.unified_step import Callsite, choose_step
from planner.tp2_lookup import Lookup


class ExportStepTest(unittest.TestCase):
    def setUp(self):
        self.runtime={'tp':2,'dtype':'bfloat16','gpu':'A800','torch':'2.7','triton':'3.4',
                      'cuda_build':'12.6','cuda_device_max_connections':'1'}
        self.route=bind({'method':'dx','actual_path':'direct_exchange','direction':'rs','fallback':False},
                        gpu='A800',topology='nv8',tp=2,dtype='bfloat16',
                        versions=(('cuda','12.6'),('torch','2.7'),('triton','3.4')),
                        timing_mode='b2b16',warmup=3,config_family='rs;connection=1')
        self.shape=(1024,5120,3072)
        self.call=Callsite('full_proj','forward_backward',self.route,self.shape,1,1.,
                           ContextRequirement(self.route,self.shape,'supported','test'),0.)
        self.registry=PredictorRegistry()
        self.registry.register(RouteRegressor(self.route).fit([
            Sample(self.route,self.shape,1.,.8), Sample(self.route,(2048,5120,3072),2.,1.6)]))
        self.cal={'workload':'qwen','common_ms':0.,'pairs_ms':{},'provenance':'test-step'}
        self.result=choose_step(self.registry,[self.call],'qwen',common_ms=0.,pairs_ms={},calibration_provenance='test-step')
        self.sources=[{'sha256':'a'*64}]

    def test_export_and_existing_runtime_loader(self):
        data=export_step_plan(self.result,[self.call],runtime=self.runtime,workload='qwen',
                              model_sources=self.sources,step_calibration=self.cal)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'plan.json';digest=write_json_new(path,data)
            lookup=Lookup(path,self.runtime)
            self.assertEqual(digest,lookup.digest)
            self.assertEqual(lookup.choose('proj',1024,3072,5120),'dx')
            self.assertEqual(lookup.choose('proj',2048,3072,5120),'stock')
            with self.assertRaises(FileExistsError):write_json_new(path,data)

    def test_context_calibration_and_training_checks(self):
        with self.assertRaises(ValueError):
            export_step_plan(self.result,[replace(self.call,phase='forward')],runtime=self.runtime,
                             workload='qwen',model_sources=self.sources,step_calibration=self.cal)
        with self.assertRaises(ValueError):
            export_step_plan(self.result,[self.call],runtime=self.runtime,workload='qwen',
                             model_sources=self.sources,step_calibration={**self.cal,'common_ms':1.})
        with self.assertRaises(ValueError):
            export_step_plan(self.result,[self.call],runtime={**self.runtime,'cuda_device_max_connections':'unset'},
                             workload='qwen',model_sources=self.sources,step_calibration=self.cal)
        bad={**self.result,'step_calibrated':False}
        with self.assertRaises(ValueError):
            export_step_plan(bad,[self.call],runtime=self.runtime,workload='qwen',
                             model_sources=self.sources,step_calibration=self.cal)


if __name__=='__main__':unittest.main()
