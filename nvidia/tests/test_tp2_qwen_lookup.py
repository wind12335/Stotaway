import json
import tempfile
import unittest
from pathlib import Path

from planner.tp2_lookup import build, Lookup, workload_signature, WORKLOAD_FIELDS
from types import SimpleNamespace


class TestQwenLookup(unittest.TestCase):
    def test_workload_signature_checked_when_profile_requires_it(self):
        args=SimpleNamespace(**{field: 1 for field in WORKLOAD_FIELDS})
        signature=workload_signature(args,local_spec=True)
        runtime={'tp':2}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'lookup.json'
            profile={'schema':'stowaway-tp2-training-lookup-v1','classification':'diagnostic-forward-backward',
                     'runtime':runtime,'entries':{},'workload_requirements':signature}
            path.write_text(json.dumps(profile))
            lookup=Lookup(path,runtime)
            lookup.validate_workload(signature)
            with self.assertRaises(ValueError):
                lookup.validate_workload({**signature,'num_layers':8})
            del profile['workload_requirements']
            path.write_text(json.dumps(profile))
            Lookup(path,runtime).validate_workload({})

    def test_shape_disambiguation_and_same_key_aggregation(self):
        runtime = {'tp': 2, 'dtype':'bfloat16', 'gpu':'A800', 'torch':'x', 'triton':'y',
                   'cuda_build':'z', 'cuda_device_max_connections':'1'}
        def row(site, op, k, n, dx):
            return {'site':site, 'op':op, 'M':1024, 'K':k, 'N':n,
                    'measurements':{'stock':{'samples_ms':[1.]*4},'dx':{'samples_ms':[dx]*4}}}
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.json';output=Path(directory)/'lookup.json'
            source.write_text(json.dumps({'metadata':{**runtime,'checks':2,
                                      'classification':'diagnostic unfused site forward+backward'},
                         'results':[row('gdn_in','ag',5120,8240,.8),row('mlp_fc1','ag',5120,17408,1.1),
                                    row('gdn_out','rs',3072,5120,.8),row('full_proj','rs',3072,5120,.8)]}))
            profile=build([source], objective='training');output.write_text(json.dumps(profile))
            lookup=Lookup(output,runtime)
            self.assertEqual(lookup.choose('fc1',1024,5120,8240),'dx')
            self.assertEqual(lookup.choose('fc1',1024,5120,17408),'stock')
            self.assertEqual(lookup.choose('fc2',1024,3072,5120),'dx')
            self.assertEqual(lookup.choose('proj',1024,3072,5120),'dx')
            self.assertEqual(lookup.choose('fc1',2048,5120,8240),'stock')


if __name__ == '__main__':
    unittest.main()
