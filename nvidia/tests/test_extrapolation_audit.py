import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from planner.original_a800.extrapolation_audit import audit


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class ExtrapolationAuditTest(unittest.TestCase):
    def test_range_and_source_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ops=root/'ops';stage=root/'stage';ops.write_text('{}');stage.write_text('{}')
            model=root/'model';decision=root/'decision';evaluation=root/'evaluation'
            model.write_text(json.dumps({'training_operator_sources':[{'path':str(ops),'sha256':sha(ops)}],
                'training_stage_sources':[{'path':str(stage),'sha256':sha(stage)}],'signature':['dx']}))
            pred={'identity':['v','fc2',1024,4096,2048],'op':'rs',
                  'query':{'M':1024,'N':4096,'K':2048,'full_ms':1.,'split_ms':1.1},
                  'prediction':{'predicted_log_speedup':.1,'choice':'overlap'}}
            decision.write_text(json.dumps({'model_sha256':sha(model),'predictions':[pred]}))
            evaluation.write_text(json.dumps({'frozen_sha256':sha(decision),'rows':[{'identity':pred['identity'],'actual_speedup':1.1}]}))
            train=[{'op':'rs','M':1024,'N':4096,'K':2048,'full_ms':1.,'split_ms':1.1}]
            with patch('planner.original_a800.extrapolation_audit.join',return_value=(train,('dx',),[],[])):
                result=audit(model,decision,[evaluation]);self.assertEqual(result['summary']['inside']['correct'],1)
                stage.write_text('changed')
                with self.assertRaisesRegex(ValueError,'source changed'):audit(model,decision,[evaluation])


if __name__=='__main__':unittest.main()
