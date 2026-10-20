import importlib.util
from pathlib import Path
import unittest
from maze.training import is_local_edit
from maze.workflow import prepare
from maze.demo import fixture
ROOT=Path(__file__).resolve().parents[1]


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class DataScriptTests(unittest.TestCase):
    def test_local_edit_chain_has_no_labels_and_is_reproducible(self):
        module=load('propose_local_edits')
        paths=module.propose(['a','b','c','d','e','f'],4,32,19)
        self.assertEqual(len(paths),32);self.assertEqual(len(set(paths)),32)
        self.assertTrue(all(is_local_edit(a,b) for a,b in zip(paths,paths[1:])))
        self.assertEqual(paths,module.propose(['a','b','c','d','e','f'],4,32,19))
    def test_benchmark_conversion(self):
        c=load('prepare_dataset').canonical
        _,m=c({'question':'Q','options':['x','y'],'answer_index':1},'mmlu_pro',1)
        self.assertEqual(m['answer'],'B')
        _,g=c({'question':'Q','answer':'Add.\n#### 42'},'gsm8k',1)
        self.assertEqual(g['answer'],'42');self.assertEqual(g['rationale'],'Add.')
        _,g=c({'Question':'Q','Correct Answer':'right','Incorrect Answer 1':'wrong1','Incorrect Answer 2':'wrong2','Incorrect Answer 3':'wrong3'},'gpqa',10)
        self.assertEqual(g['choices'][ord(g['answer'])-65],'right')
    def test_explicit_training_pools_are_executed_without_scorer(self):
        b,q,_,_=fixture();q={'q8':q['q8']}
        module=load('propose_local_edits');paths={'q8':module.propose(list(b),4,8,4)}
        config={'model':'mock','model_revision':'1','policy':'topk','calls':8,'shots':4,'pool_size':8,'seed':1,
                'decoding':{'temperature':.3,'max_tokens':20},'tie_rule':'first','selection':'uniform','purpose':'offline_training'}
        plan=prepare(b,q,'S',config,paths_by_query=paths)
        self.assertEqual({tuple(r['path']) for r in plan['requests']},set(paths['q8']))
