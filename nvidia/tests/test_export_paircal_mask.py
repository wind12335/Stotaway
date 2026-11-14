import unittest

from planner.original_a800.export_paircal_mask import CONTEXT, INVENTORY, RUNTIME, export
from planner.tp2_lookup import TRAIN_SCHEMA


class ExportPaircalTest(unittest.TestCase):
    def test_masks_and_shape_restriction(self):
        report = {'context': CONTEXT, 'blocks': [
            {'block': i, 'median_delta_ms': {'stock': 0, 'proj': 1, 'fc2': -2, 'pair': -1}}
            for i in range(3)]}
        inventory = {'schema': TRAIN_SCHEMA, 'runtime': RUNTIME.copy(),
                     'entries': {k: {} for k in INVENTORY} | {'fc2:4096:4:6': {}}}
        plan=export(report,inventory,report_sha='a',inventory_sha='b')
        self.assertEqual(plan['selected_mask'],'fc2')
        self.assertEqual(plan['entries']['fc2:2048:3072:5120']['choice'],'dx')
        self.assertEqual(plan['entries']['proj:2048:3072:5120']['choice'],'stock')
        self.assertNotIn('fc2:4096:4:6',plan['entries'])
        inventory['runtime']['gpu']='other'
        with self.assertRaises(ValueError):export(report,inventory,report_sha='a',inventory_sha='b')
        inventory['runtime']=RUNTIME.copy()
        report['context']='other'
        with self.assertRaises(ValueError):export(report,inventory,report_sha='a',inventory_sha='b')


if __name__=='__main__':unittest.main()
