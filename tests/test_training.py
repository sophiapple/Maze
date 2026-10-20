from pathlib import Path
import importlib.util
import tempfile
import unittest
import numpy as np
from maze.demo import fixture
from maze.training import observations,is_local_edit
from maze.calibration import fit_residual,predict_residual


class TrainingContracts(unittest.TestCase):
    def test_local_edit_definition(self):
        self.assertTrue(is_local_edit(('a','b','c'),('a','d','c')))
        self.assertTrue(is_local_edit(('a','b','c'),('b','a','c')))
        self.assertFalse(is_local_edit(('a','b','c'),('c','a','b')))
    def test_test_labels_never_enter_fitting(self):
        _,_,store,_=fixture()
        row={'query_id':'q0','path':['e1','e2','e3','e4'],'target':1,'split':'test'}
        with self.assertRaises(ValueError):observations([row],store,4)
    def test_question_split_overlap_rejected(self):
        _,_,store,_=fixture()
        row={'query_id':'q0','path':['e1','e2','e3','e4'],'target':1,'split':'train'}
        with self.assertRaises(ValueError):observations([row,row|{'split':'validation'}],store,4)
    def test_residual_zero_anchor_and_heldout_isolation(self):
        model=fit_residual([[0,0],[1,2],[2,1],[0,0]],[0,.5,.25,0],['a','a','b','b'],held_out_keys=['c'])
        self.assertEqual(predict_residual(model,[[0,0]])[0],0.)
        with self.assertRaises(ValueError):fit_residual([[1],[2]],[1,0],['a','b'],held_out_keys=['a'])
    def test_residual_matches_preserved_source(self):
        path=Path(__file__).resolve().parents[1]/'reproduction/compact/research/energy_residual_v142.py'
        spec=importlib.util.spec_from_file_location('legacy_residual',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rng=np.random.default_rng(14);x=rng.normal(size=(8,4));y=rng.uniform(-1,1,size=8);keys=['a']*4+['b']*4
        old=module.fit_residual(x,y,keys);new=fit_residual(x,y,keys)
        np.testing.assert_allclose(new['beta'],old['beta'],atol=1e-12)
        np.testing.assert_allclose(predict_residual(new,x),module.residual_scores(old,x),atol=1e-12)


@unittest.skipUnless(importlib.util.find_spec('torch'),'Install the train extra')
class TrainingIntegration(unittest.TestCase):
    def test_train_export_and_input_only_inference(self):
        from maze.demo import train_demo
        from maze.io import read_json
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'fit';result=train_demo(out)
            self.assertGreater(result['best_epoch'],0)
            metadata=read_json(out/'checkpoint/checkpoint.json')['metadata']
            self.assertEqual(len(metadata['train_question_keys']),6)
            self.assertEqual(len(metadata['validation_question_keys']),2)
            self.assertFalse(set(metadata['train_question_keys'])&set(metadata['validation_question_keys']))
            self.assertFalse(metadata['auxiliary_heads_trained'])
            plan=read_json(out/'held_input_plan.json')
            self.assertEqual(plan['planning_target_calls'],0)
