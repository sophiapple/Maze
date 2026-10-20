"""Portable gold-blind numerical replay of archived question-node energy heads.

No training, dataset/model download, target generation, grading or paid API.
Caller supplies an independently verified input/checkpoint bundle under its terms.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,time,sys
import numpy as np
CONDITION_FIELDS={'index','query_id','path','condition_key','nodes','anchor'}
GROUP_FIELDS={'indices','anchor'}
ARCH_FIELDS={'dim','views','rank','hidden','start_mode'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def member(root,rel):
    root=Path(root).resolve();p=(root/rel).resolve()
    if root not in p.parents or not p.is_file():raise ValueError('Only regular files inside bundle accepted')
    return p

def validate_input(m):
    required={'benchmark','p','target','system_mode','outer_folds','groups','conditions','bank_hash'}
    expected=required|({'system_receipt'} if m.get('system_mode')=='encoded_system_full_qa' else set())
    if set(m)!=expected:raise ValueError('Input manifest must exclude labels/observations/costs')
    if m['p']!=4 or m['target']['thinking'] is not False or m['system_mode'] not in {'zero_start_question_nodes','encoded_system_full_qa'}:raise ValueError('Wrong archived policy/input mode')
    if set(m['outer_folds'])!=set(m['groups']):raise ValueError('Unmatched frozen question membership')
    if len(m['conditions'])!=9*len(m['groups']):raise ValueError('Nine frozen p4 conditions per query required')
    if len({c['condition_key'] for c in m['conditions']})!=len(m['conditions']):raise ValueError('Repeated condition identity')
    for j,c in enumerate(m['conditions']):
        if set(c)!=CONDITION_FIELDS or c['index']!=j or len(c['path'])!=4 or len(set(c['path']))!=4 or len(c['nodes'])!=6:
            raise ValueError('Strict input-only condition schema/p4 required')
    covered=[]
    for q,g in m['groups'].items():
        if set(g)!=GROUP_FIELDS or len(g['indices'])!=9 or len(set(g['indices']))!=9 or g['anchor'] not in g['indices']:raise ValueError('Invalid frozen group')
        covered.extend(g['indices'])
        for i in g['indices']:
            c=m['conditions'][i]
            if c['query_id']!=q or c['anchor']!=g['anchor']:raise ValueError('Invalid fixed retrieval anchor')
    if sorted(covered)!=list(range(len(m['conditions']))):raise ValueError('Every input condition must be used exactly once')
    return m

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def replay(bundle):
    import torch
    torch.set_num_threads(2)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():raise ValueError('Replay uses CPU only')
    root=Path(bundle);index=read(root/'bundle.json')
    family=index['policy_family']
    families={'v142_question_calibrated_oof','v118_question_anchor_harm_oof','v126_fullqa_anchor_harm_oof'}
    if family not in families:raise ValueError('Unsupported frozen policy family')
    referenced={index['inputs'],index['nodes']}
    for entry in index['folds'].values():
        referenced.update([entry['metadata'],entry['weights']])
        if family=='v142_question_calibrated_oof':referenced.add(entry['calibration'])
    if not referenced.issubset(index['files']):raise ValueError('Every referenced input/array/checkpoint/head requires a hash')
    for value in index['files'].values():
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Complete lowercase SHA256 identities required')
    for rel,h in index['files'].items():
        if sha(member(root,rel))!=h:raise ValueError('Bundle identity mismatch')
    sources=Path(__file__).resolve().parent/'research'
    expected_sources={'structured_energy_v118.py'}
    if family=='v142_question_calibrated_oof':expected_sources.add('energy_residual_v142.py')
    if family=='v126_fullqa_anchor_harm_oof':expected_sources.add('structured_energy_v126_portable.py')
    if set(index['source_hashes'])!=expected_sources:raise ValueError('Only reviewed source modules accepted')
    for name,h in index['source_hashes'].items():
        if sha(sources/name)!=h:raise ValueError('Archived scorer source mismatch')
    geometry=load_module('maze_replay_geometry',sources/'structured_energy_v118.py')
    residual=load_module('maze_replay_residual',sources/'energy_residual_v142.py') if family=='v142_question_calibrated_oof' else None
    if family=='v126_fullqa_anchor_harm_oof':
        full=load_module('maze_replay_fullqa',sources/'structured_energy_v126_portable.py')
        Model=full.StructuredEnergyV126
    else:Model=geometry.StructuredEnergyV118
    m=validate_input(read(member(root,index['inputs'])))
    table=np.load(member(root,index['nodes']),allow_pickle=False)
    views=32 if family=='v126_fullqa_anchor_harm_oof' else 4
    if table.dtype!=np.float16 or table.ndim!=3 or table.shape[1:]!=(views,2560) or not np.isfinite(table).all():raise ValueError('Wrong canonical node array')
    if any(min(c['nodes'])<0 or max(c['nodes'])>=len(table) for c in m['conditions']):raise ValueError('Node index out of bounds')
    if family=='v126_fullqa_anchor_harm_oof':
        if m['system_mode']!='encoded_system_full_qa' or not m['system_receipt'].get('exact_system_input_verified') or not m['system_receipt'].get('canonical_layer32_finalnorm'):raise ValueError('Encoded system/canonical recipe identity required')
        if any(not np.any(table[c['nodes'][0]]) for c in m['conditions']):raise ValueError('Actual encoded system required')
    elif m['system_mode']!='zero_start_question_nodes' or any(np.any(table[c['nodes'][0]]) for c in m['conditions']):raise ValueError('Archived zero-start input must be zero')
    used={j for c in m['conditions'] for j in c['nodes'][1:]}
    if any(np.any(np.linalg.norm(table[j].astype(np.float64),axis=-1)<=0) for j in used):raise ValueError('Every query/exemplar view must be nonzero')
    result={};begin=time.perf_counter()
    for fold,entry in sorted(index['folds'].items(),key=lambda x:int(x[0])):
        info=read(member(root,entry['metadata']));architecture=info['architecture']
        expected_arch={'dim':2560,'views':32,'rank':16,'hidden':32,'start_mode':'encoded_system_full_qa'} if family=='v126_fullqa_anchor_harm_oof' else {'dim':2560,'views':4,'rank':64,'hidden':96,'start_mode':'zero_start_question_nodes'}
        if set(architecture)!=ARCH_FIELDS or architecture!=expected_arch:raise ValueError('Wrong frozen architecture')
        if info['bank_hash']!=m['bank_hash'] or info['outer_fold']!=int(fold):raise ValueError('Checkpoint/input identity mismatch')
        held=sorted(q for q,f in m['outer_folds'].items() if f==int(fold))
        if held!=info['held_query_ids'] or set(held)&set(info['calibration_query_ids']):raise ValueError('Checkpoint/fold membership mismatch')
        model=Model(**architecture)
        with np.load(member(root,entry['weights']),allow_pickle=False) as arrays:
            state={k:torch.from_numpy(arrays[k].copy()) for k in arrays.files}
        model.load_state_dict(state,strict=True);model.eval()
        ids=[i for q in held for i in m['groups'][q]['indices']];pos={i:j for j,i in enumerate(ids)}
        nodes=torch.tensor([m['conditions'][i]['nodes'] for i in ids],dtype=torch.long)
        anchors=torch.tensor([pos[m['conditions'][i]['anchor']] for i in ids],dtype=torch.long)
        with torch.no_grad():
            mu,mass=model.project(torch.as_tensor(table.astype(np.float32)))
            output=model.forward_projected(mu,mass,nodes,anchors,**({'system_receipt':m['system_receipt']} if family=='v126_fullqa_anchor_harm_oof' else {}))
        head=read(member(root,entry['calibration'])) if residual else None
        for q in held:
            g=m['groups'][q];forecasts={}
            for key,source in [('scores','logits'),('uplift','uplift'),('harm','harm_probability')]:
                forecasts[key]={i:float(output[source][pos[i]]) for i in g['indices']}
            keys={i:m['conditions'][i]['condition_key'] for i in g['indices']}
            if residual:
                x=residual.features([forecasts],g['indices'],g['anchor']);utility=residual.residual_scores(head,x)
                policies={policy:residual.choose(utility,g['indices'],g['anchor'],keys,policy) for policy in ['one_call','top3','anchor3']}
                extra={'residual_utilities':dict(zip(map(str,g['indices']),map(float,utility)))}
            else:
                policies={policy:geometry.select_policy(forecasts['scores'],g['anchor'],keys,policy,forecasts['uplift'],forecasts['harm']) for policy in ['one_call','top3','anchor3','harm_guard']}
                extra={}
            result[q]=dict(fold=int(fold),source_scores={k:{str(i):v for i,v in values.items()} for k,values in forecasts.items()},policies=policies,**extra)

    if set(result)!=set(m['groups']):raise ValueError('Incomplete checkpoint fold set')
    return dict(state='complete_gold_blind_replay',policy_family=index['policy_family'],questions=len(result),p=4,decisions=result,
        wall_seconds=time.perf_counter()-begin,new_fits=0,new_target_api_encoder_calls=0,
        bundle_sha256=sha(root/'bundle.json'),claims='Numerical replay from supplied input states; not new benchmark outcomes or full data/encoder/target reconstruction')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    output=Path(args.output)
    if output.exists():raise SystemExit('Refusing to overwrite an existing replay')
    value=replay(args.bundle);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k!='decisions'}))
