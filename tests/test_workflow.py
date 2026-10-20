import copy
import tempfile
from pathlib import Path
import unittest
import numpy as np
from maze.demo import fixture,ArithmeticTarget
from maze.data import Query,Exemplar,render_messages,question_key
from maze.features import FeatureStore
from maze.planning import Candidate,allocate,sample_paths,vote
from maze.workflow import prepare,collect,grade,extract_answer,validate_plan
from maze.backends.chat_api import ChatAPI,Generation
from maze.io import read_jsonl,read_json,safe_member
from maze.scorers.numpy_energy import EnergyConfig,NumpyEnergy


class InputTests(unittest.TestCase):
    def test_normalized_question_key(self):
        self.assertEqual(question_key('  WHAT\t is 2+2? '),question_key('what is 2+2?'))
    def test_query_rejects_answers(self):
        with self.assertRaises(ValueError):Query.from_dict({'id':'q','question':'x','answer':'A'})
    def test_complete_rendering(self):
        b={'e':Exemplar('e','Train?','A','Full rationale here.')};q=Query('q','Test?',('one','two'))
        text=render_messages('System',b,('e',),q)[1]['content']
        self.assertIn('Full rationale here.',text);self.assertIn('Final answer: A',text);self.assertIn('B. two',text)
    def test_self_exemplar_rejected(self):
        b={'e':Exemplar('e','Same?','A')}
        with self.assertRaises(ValueError):render_messages('S',b,('e',),Query('q','same?'))
    def test_feature_roundtrip_and_metadata(self):
        b,q,store,_=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'features';store.save(p);loaded=FeatureStore.load(p)
            np.testing.assert_array_equal(store.states,loaded.states)
            with self.assertRaises(ValueError):safe_member(p,'../escape')
        with self.assertRaises(ValueError):store.path_indices('q1',[('e1','e1','e2','e3')])


class AllocationTests(unittest.TestCase):
    def setUp(self):self.pool=[Candidate(('e1','e2'),2.),Candidate(('e2','e1'),1.),Candidate(('e3','e1'),0.)]
    def test_independent_duplicate_winners_still_have_calls(self):
        slots=allocate([self.pool]*3,'independent',3,seed=7)
        self.assertEqual(len(slots),3);self.assertEqual(len({s.path for s in slots}),1)
        self.assertEqual(len({s.seed for s in slots}),3)
    def test_repeat_is_general_allocation_special_case(self):
        a=allocate([self.pool],'repeat',3,seed=7)
        b=allocate([self.pool],'allocation',3,seed=7,repetitions=[3])
        self.assertEqual(a,b)
    def test_mixed_allocation(self):
        a=allocate([self.pool],'allocation',4,repetitions=[2,1,1])
        self.assertEqual([s.path for s in a],[self.pool[0].path,self.pool[0].path,self.pool[1].path,self.pool[2].path])
    def test_unique_topk(self):
        self.assertEqual(len({s.path for s in allocate([self.pool],'topk',3)}),3)
    def test_invalid_budgets(self):
        for policy,calls in [('topk',4),('repeat',0),('independent',2)]:
            with self.assertRaises(ValueError):allocate([self.pool],policy,calls)
    def test_sampling_exact_and_reproducible(self):
        a=sample_paths(['a','b','c'],2,6,19)
        self.assertEqual(a,sample_paths(['c','b','a'],2,6,19));self.assertEqual(len(set(a)),6)
        with self.assertRaises(ValueError):sample_paths(['a','b'],2,3,0)
    def test_vote_counts_repeated_answers(self):
        self.assertEqual(vote(['A','B','B',None]),'B');self.assertIsNone(vote([None,'']))
        self.assertEqual(vote(['B','A'],tie='first'),'B')
        self.assertEqual(vote(['B','A'],seed=7),vote(['B','A'],seed=7))


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.bank,self.queries,self.features,rng=fixture()
        c=EnergyConfig('paper_scalar',8,7,3,5)
        self.scorer=NumpyEnergy(c,{k:rng.normal(scale=.1,size=s) for k,s in c.score_shapes.items()})
        self.config={'model':'test','model_revision':'test-1','policy':'repeat','calls':3,'shots':4,'pool_size':10,
                     'seed':8,'decoding':{'temperature':.3,'max_tokens':64},'tie_rule':'first'}
    def plan(self,selection='energy'):
        cfg=self.config|{'selection':selection}
        if selection=='retrieval':cfg['pool_size']=1
        return prepare(self.bank,{'q8':self.queries['q8']},'Solve.',cfg,features=self.features,scorer=self.scorer)
    def test_input_only_plans_and_separate_seeds(self):
        plan=self.plan();validate_plan(plan)
        self.assertEqual(len(plan['requests']),3)
        self.assertEqual(len({r['request_id'] for r in plan['requests']}),3)
        self.assertEqual(len({str(r['path']) for r in plan['requests']}),1)
        self.assertEqual(plan['planning_target_calls'],0)
    def test_alternative_baselines_execute(self):
        for selection in ('uniform','retrieval'):validate_plan(self.plan(selection))
    def test_checkpoint_question_overlap_rejected(self):
        with self.assertRaises(ValueError):prepare(self.bank,{'q8':self.queries['q8']},'S',self.config,
            features=self.features,scorer=self.scorer,checkpoint_metadata={'train_question_keys':[self.queries['q8'].key]})
    def test_modified_plan_rejected(self):
        plan=self.plan();plan['requests'][0]['seed']+=1
        with self.assertRaises(ValueError):validate_plan(plan)
    def test_collector_resume_never_repeats_completed_call(self):
        class Count(ArithmeticTarget):
            def __init__(self):self.calls=0
            def generate(self,request):self.calls+=1;return super().generate(request)
        plan=self.plan();target=Count()
        with tempfile.TemporaryDirectory() as tmp:
            collect(plan,target,tmp);collect(plan,target,tmp);self.assertEqual(target.calls,3)
            report=grade(plan,read_jsonl(Path(tmp)/'events.jsonl'),[{'query_id':'q8','answer':'57'}],kind='numeric')
            self.assertEqual(report['accuracy_exact'],'1');self.assertIsNone(report['outcomes'][0]['processed_tokens'])
    def test_uncertain_physical_request_is_not_retried(self):
        class Timeout:
            calls=0
            def generate(self,request):self.calls+=1;raise TimeoutError()
        plan=self.plan();target=Timeout()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):collect(plan,target,tmp)
            with self.assertRaises(RuntimeError):collect(plan,target,tmp)
            self.assertEqual(target.calls,1)
    def test_invalid_and_truncated_responses_stay_charged(self):
        class Invalid:
            def generate(self,request):return Generation('cannot parse',100,20,'length')
        plan=self.plan()
        with tempfile.TemporaryDirectory() as tmp:
            collect(plan,Invalid(),tmp)
            report=grade(plan,read_jsonl(Path(tmp)/'events.jsonl'),[{'query_id':'q8','answer':'A'}],kind='choice')
            row=report['outcomes'][0]
            self.assertEqual((row['calls'],row['invalid_calls'],row['truncated_calls']),(3,3,3))
            self.assertEqual(row['input_tokens'],300);self.assertEqual(row['correct'],0)
    def test_partial_evaluation_rejected(self):
        with self.assertRaises(ValueError):grade(self.plan(),[],[{'query_id':'q8','answer':'A'}],kind='choice')
    def test_explicit_answer_parsers(self):
        self.assertEqual(extract_answer('Reasoning\nFinal answer: B','choice'),'B')
        self.assertEqual(extract_answer('Final answer: 0.5','numeric'),'1/2')
        self.assertIsNone(extract_answer('Reasoning mentions A and B','choice'))
        self.assertIsNone(extract_answer("__import__('os')",'numeric'))
    def test_api_security_and_accounting(self):
        for url in ('http://remote.example/v1','https://name:secret@example.org/v1','https://example.org/v1?key=secret'):
            with self.assertRaises(ValueError):ChatAPI(url)
        ChatAPI('http://127.0.0.1:8000/v1')
        with self.assertRaises(ValueError):Generation('text',-1,0,'stop')
