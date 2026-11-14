import unittest

from planner.original_a800.step_mask_calibration import StepBlock, select


class StepMaskCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.context = 'dx-sm80-tp2-conn1-workload'
        self.blocks = [StepBlock(self.context, '1', {'stock': 100, 'a': 95, 'b': 103}),
                       StepBlock(self.context, '2', {'stock': 110, 'a': 105, 'b': 113})]

    def test_within_block_differences_cancel_shared_drift(self):
        result = select(self.blocks, context=self.context, allowed_masks={'stock', 'a', 'b'})
        self.assertEqual(result['choice'], 'a')
        self.assertEqual(result['masks']['a']['deltas_ms'], [-5, -5])

    def test_unstable_and_missing_evidence_falls_back(self):
        blocks = [StepBlock(self.context, '1', {'stock': 100, 'a': 90}),
                  StepBlock(self.context, '2', {'stock': 100, 'a': 105})]
        self.assertEqual(select(blocks, context=self.context, allowed_masks={'stock', 'a'})['choice'], 'stock')
        self.assertEqual(select(blocks[:1], context=self.context, allowed_masks={'stock', 'a'})['choice'], 'stock')
        with self.assertRaises(ValueError):
            select(blocks, context='other', allowed_masks={'stock', 'a'})
        with self.assertRaises(ValueError):
            select(blocks, context=self.context, allowed_masks={'stock', 'a', 'b'})
        with self.assertRaises(ValueError):
            select([blocks[0], blocks[0]], context=self.context, allowed_masks={'stock', 'a'})

    def test_invalid_time(self):
        with self.assertRaises(ValueError):
            StepBlock(self.context, '1', {'stock': 100, 'a': float('nan')})


if __name__ == '__main__':
    unittest.main()
