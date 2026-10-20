"""Synthetic integration fixtures, deliberately separate from measured results."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import numpy as np
from maze.backends.chat_api import Generation
from maze.data import Query, Exemplar
from maze.features import FeatureStore
from maze.io import write_json, write_jsonl, read_jsonl, identity
from maze.planning import sample_paths
from maze.scorers.numpy_energy import EnergyConfig,NumpyEnergy,save_checkpoint,load_checkpoint
from maze.workflow import prepare,collect,grade


class ArithmeticTarget:
    """A deterministic Python stub, not an LLM or an empirical baseline."""
    def generate(self,request):
        import re
        a,b=map(int,re.findall(r'What is (\d+) plus (\d+)\?',request['messages'][-1]['content'])[-1])
        return Generation(f'Final answer: {a+b}',None,None,'stop','mock-arithmetic')


def fixture():
    bank={f'e{i}':Exemplar(f'e{i}',f'What is {i} plus {i+2}?',str(2*i+2),
                          f'Add {i} and {i+2}.') for i in range(1,8)}
    queries={f'q{i}':Query(f'q{i}',f'What is {i+20} plus {i+21}?') for i in range(9)}
    nodes=[{'id':'__system__','kind':'system','question_key':None}]
    nodes += [{'id':e.id,'kind':'exemplar','question_key':e.key} for e in bank.values()]
    nodes += [{'id':q.id,'kind':'query','question_key':q.key} for q in queries.values()]
    rng=np.random.default_rng(20261020)
    states=rng.normal(size=(len(nodes),7,8)).astype(np.float32)
    recipe={'model':'synthetic-fixture','revision':'1','layers':list(range(7)),
            'pooling':'invented-vectors','start_mode':'encoded_system_full_qa',
            'node_encoding':'synthetic','role':'software_test_not_benchmark'}
    return bank,queries,FeatureStore(states,nodes,recipe),rng


def run_demo(output:str|Path):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    bank,queries,features,rng=fixture()
    config=EnergyConfig('paper_scalar',dim=8,views=7,rank=3,hidden=5,shots=4)
    weights={k:rng.normal(scale=.15,size=s) for k,s in config.score_shapes.items()}
    save_checkpoint(root/'checkpoint',config,weights,{'role':'synthetic_untrained_fixture'})
    features.save(root/'features')
    scorer=NumpyEnergy(config,weights)
    selected={qid:q for qid,q in queries.items() if qid in {'q7','q8'}}
    policy={'model':'mock-arithmetic','model_revision':'1','policy':'repeat','calls':3,
            'shots':4,'pool_size':20,'seed':73,'decoding':{'temperature':.3,'max_tokens':128},'tie_rule':'first'}
    plan=prepare(bank,selected,'Solve the arithmetic problem. End with Final answer: ...',policy,
                 features=features,scorer=scorer)
    write_json(root/'plan.json',plan)
    collect(plan,ArithmeticTarget(),root/'collection')
    answers=[{'query_id':f'q{i}','answer':str(41+2*i)} for i in (7,8)]
    report=grade(plan,read_jsonl(root/'collection/events.jsonl'),answers,kind='numeric')
    report['role']='synthetic_software_test_not_benchmark'
    write_json(root/'evaluation.json',report)
    write_jsonl(root/'bank.jsonl',[asdict(e) for e in bank.values()])
    write_jsonl(root/'queries.jsonl',[asdict(q) for q in selected.values()])
    return {'role':'synthetic software demo, not a benchmark result','queries':len(selected),
            'target':'deterministic Python stub','requests':len(plan['requests']),'output':str(root)}


def train_demo(output:str|Path):
    from maze.training import fit
    root=Path(output)
    if root.exists():raise FileExistsError(root)
    bank,queries,features,rng=fixture()
    config=EnergyConfig('paper_scalar',dim=8,views=7,rank=3,hidden=5,shots=4)
    teacher=NumpyEnergy(config,{k:rng.normal(scale=.25,size=s) for k,s in config.score_shapes.items()})
    observations=[]
    for i in range(8):
        qid=f'q{i}';paths=sample_paths(bank,4,10,100+i)
        scores=teacher.score(features.states,features.path_indices(qid,paths))
        targets=(scores>np.median(scores)).astype(int)
        for j,(path,y) in enumerate(zip(paths,targets)):
            observations.append({'query_id':qid,'path':list(path),'target':int(y),
                                 'split':'train' if i<6 else 'validation','reference':j==0})
    root.mkdir(parents=True)
    features.save(root/'features');write_jsonl(root/'observations.jsonl',observations)
    result=fit(features,observations,config,{'epochs':12,'patience':4,'pair_mode':'all_preferences',
               'learning_rate':.005,'seed':20261020},root/'checkpoint')
    model,metadata=load_checkpoint(root/'checkpoint')
    eval_queries={'q8':queries['q8']}
    policy={'model':'mock-arithmetic','model_revision':'1','policy':'repeat','calls':1,
            'shots':4,'pool_size':20,'seed':400,'decoding':{'temperature':.3,'max_tokens':128},'tie_rule':'first'}
    plan=prepare(bank,eval_queries,'End with Final answer: ...',policy,features=features,scorer=model,
                 checkpoint_metadata=metadata['metadata'])
    write_json(root/'held_input_plan.json',plan)
    result['role']='synthetic gradient/checkpoint integration test; no empirical benchmark claim'
    write_json(root/'training_demo.json',result)
    return result
