import tempfile
import unittest
from pathlib import Path

from planner.original_a800.report_qwen36_tp2_e2e import parse


class ReportQwenTest(unittest.TestCase):
    def test_rejects_missing_or_nonfinite_loss(self):
        with tempfile.TemporaryDirectory() as root:
            log=Path(root)/'run.log'
            log.write_text('iteration 4/10 | elapsed time per iteration (ms): 2.0 | lm loss: nan\n[after training is done]')
            with self.assertRaises(ValueError):parse(log,3)
            log.write_text('iteration 4/10 | elapsed time per iteration (ms): 2.0 | lm loss: 1.0\n[after training is done]')
            self.assertEqual(parse(log,3)['after_warmup_median_ms'],2.)


if __name__=='__main__':unittest.main()
