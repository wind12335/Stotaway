import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from planner.original_a800.audit_plan_sources import audit


def ref(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


class SourceAuditTest(unittest.TestCase):
    def test_detect_modified_source_and_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'ops';b=root/'stage';c=root/'target'
            for p in (a,b,c):p.write_text('{}')
            model=root/'model.json';plan=root/'plan.json'
            model.write_text(json.dumps({'schema':'stowaway-fragmentation-freeze-v1',
                'training_operator_sources':[ref(a)],'training_stage_sources':[ref(b)],
                'targets':[{'model':'llama','M':1024}]}))
            plan.write_text(json.dumps({'plan_source':'reduced-probe-model','phase':'forward',
                'model_sha256':ref(model)['sha256'],'target_model':'llama',
                'workload_requirements':{'seq_length':1024,'micro_batch_size':1},'component_sources':[ref(c)]}))
            self.assertEqual(len(audit(plan,model)['verified_sources']),3)
            b.write_text('changed')
            with self.assertRaisesRegex(ValueError,'source file hash mismatch'):audit(plan,model)
            model.write_text('changed')
            with self.assertRaises((ValueError,json.JSONDecodeError)):audit(plan,model)


if __name__=='__main__':unittest.main()
