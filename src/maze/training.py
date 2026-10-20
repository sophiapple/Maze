"""Train only the compact planner from frozen states and training observations."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import random
import numpy as np
from maze.features import FeatureStore
from maze.io import identity, read_jsonl, write_json
from maze.scorers.numpy_energy import EnergyConfig, save_checkpoint


def is_local_edit(a:tuple[str,...],b:tuple[str,...]) -> bool:
    if len(a)!=len(b):return False
    changed=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
    if len(changed)==1:return len(set(a)&set(b))==len(a)-1
    if len(changed)==2:
        i,j=changed
        return a[i]==b[j] and a[j]==b[i]
    return False


def observations(rows:list[dict],store:FeatureStore,shots:int) -> dict:
    groups=defaultdict(list);ids=set();membership={}
    for row in rows:
        if not {'query_id','path','target','split'}<=set(row) or not set(row)<={'query_id','path','target','split','reference'}:
            raise ValueError('Observation fields are query_id, path, target, split and optional reference')
        if row['split'] not in {'train','validation'}:
            raise ValueError('Only training/validation observations may enter fitting')
        y=row['target']
        if isinstance(y,bool) or not isinstance(y,(float,int)) or not np.isfinite(y) or not 0<=y<=1:
            raise ValueError('Correctness target must lie in [0,1]')
        if len(row['path'])!=shots or ('reference' in row and type(row['reference']) is not bool):
            raise ValueError('Invalid observation path or reference flag')
        qid=row['query_id'];node=store.nodes[store.lookup[qid]]
        key=node['question_key']
        if key in membership and membership[key]!=row['split']:
            raise ValueError('Normalized question overlaps fitting and model selection')
        membership[key]=row['split']
        idx=store.path_indices(qid,[row['path']])[0]
        uid=(qid,tuple(row['path']))
        if uid in ids:raise ValueError('Aggregate repeated responses before fitting; duplicate path record')
        ids.add(uid)
        groups[row['split'],qid].append(dict(row,indices=idx))
    if not groups or not any(s=='train' for s,_ in groups) or not any(s=='validation' for s,_ in groups):
        raise ValueError('Separate nonempty training and validation queries are required')
    val={k for k,v in membership.items() if v=='validation'}
    if val&{n['question_key'] for n in store.nodes if n['kind']=='exemplar'}:
        raise ValueError('Validation questions must not appear in the training exemplar bank')
    return dict(groups)


def fit(store:FeatureStore,rows:list[dict],architecture:EnergyConfig,options:dict,output:str|Path) -> dict:
    import torch
    from torch.nn import functional as F
    from maze.scorers.torch_energy import GeometricEnergy
    defaults={'seed':20261020,'epochs':40,'learning_rate':1e-3,'weight_decay':1e-3,
              'patience':5,'rank_weight':0.5,'success_weight':1.0,'pair_mode':'reference',
              'gradient_clip':1.0,'device':'cpu'}
    if not set(options)<=set(defaults):raise ValueError('Unrecognized training option')
    cfg=defaults|options
    if any(type(cfg[k]) is not int or cfg[k]<1 for k in ['epochs','patience']):
        raise ValueError('epochs and patience must be positive integers')
    if type(cfg['seed']) is not int or cfg['pair_mode'] not in {'reference','all_preferences','local_edits'}:
        raise ValueError('Invalid seed or pair construction')
    for key in ['learning_rate','weight_decay','rank_weight','success_weight','gradient_clip']:
        value=cfg[key]
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not np.isfinite(value) or value<0:
            raise ValueError('Training hyperparameters must be finite nonnegative numbers')
    if cfg['learning_rate']==0 or cfg['gradient_clip']==0 or cfg['rank_weight']+cfg['success_weight']==0:
        raise ValueError('Training must have positive rate, gradient cap and objective weight')
    if store.states.shape[1:]!=(architecture.views,architecture.dim) or store.recipe['start_mode']!=architecture.start_mode:
        raise ValueError('Architecture does not match frozen encoder features')
    root=Path(output)
    if root.exists():raise FileExistsError(root)
    groups=observations(rows,store,architecture.shots)
    torch.manual_seed(cfg['seed']);rng=random.Random(cfg['seed'])
    model=GeometricEnergy(architecture).to(cfg['device'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    tensors={}
    for key,records in groups.items():
        x=torch.as_tensor(store.states[np.stack([r['indices'] for r in records])].astype(np.float32),device=cfg['device'])
        y=torch.tensor([r['target'] for r in records],dtype=torch.float32,device=cfg['device'])
        if cfg['pair_mode']=='reference':
            anchors=[i for i,r in enumerate(records) if r.get('reference',False)]
            if len(anchors)!=1:raise ValueError('Each query needs exactly one fixed reference prompt')
            pairs=[(i,anchors[0]) for i in range(len(records)) if i!=anchors[0]]
        else:
            pairs=[(i,j) for i in range(len(records)) for j in range(len(records))
                   if records[i]['target']>records[j]['target'] and
                   (cfg['pair_mode']!='local_edits' or is_local_edit(tuple(records[i]['path']),tuple(records[j]['path'])))]
        tensors[key]=(x,y,pairs)
    if cfg['success_weight']==0 and not any(pairs for (s,_),(x,y,pairs) in tensors.items() if s=='train'):
        raise ValueError('No informative pairs for the requested pair-only training objective')
    def loss(key):
        x,y,pairs=tensors[key]
        logits=model(x)['logits']
        success=F.binary_cross_entropy_with_logits(logits,y)
        if pairs:
            i,j=zip(*pairs)
            differences=logits[list(i)]-logits[list(j)]
            if cfg['pair_mode']=='reference':
                pair_targets=(1+y[list(i)]-y[list(j)])/2
                ranking=F.binary_cross_entropy_with_logits(differences,pair_targets)
            else:ranking=F.softplus(-differences).mean()
        else:ranking=logits.sum()*0
        return cfg['success_weight']*success+cfg['rank_weight']*ranking
    train=[k for k in tensors if k[0]=='train'];val=[k for k in tensors if k[0]=='validation']
    best=float('inf');best_state=None;best_epoch=0;history=[];wait=0
    for epoch in range(1,cfg['epochs']+1):
        model.train();rng.shuffle(train);values=[]
        for key in train:
            optimizer.zero_grad(set_to_none=True);value=loss(key)
            if not torch.isfinite(value):raise ValueError('Nonfinite training loss')
            value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
            optimizer.step();values.append(float(value.detach().cpu()))
        model.eval()
        with torch.no_grad():validation=sum(float(loss(key).cpu()) for key in val)/len(val)
        if not np.isfinite(validation):raise ValueError('Nonfinite validation loss')
        history.append({'epoch':epoch,'train_loss':sum(values)/len(values),'validation_loss':validation})
        if validation<best-1e-8:
            best=validation;best_epoch=epoch;wait=0
            best_state={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
        else:wait+=1
        if wait>=cfg['patience']:break
    # Auxiliary heads are not fitted by this success/ranking trainer; do not export them.
    weights={k:v for k,v in best_state.items() if k in architecture.score_shapes}
    metadata={'train_question_keys':sorted({store.nodes[store.lookup[q]]['question_key'] for s,q in groups if s=='train'}),
              'validation_question_keys':sorted({store.nodes[store.lookup[q]]['question_key'] for s,q in groups if s=='validation'}),
              'training_options':cfg,'best_epoch':best_epoch,'best_validation_loss':best,
              'feature_recipe':store.recipe,'auxiliary_heads_trained':False,
              'training_role':'user_supplied_frozen_feature_training','observation_digest':identity(rows),
              'torch_version':torch.__version__,'numpy_version':np.__version__}
    save_checkpoint(root,architecture,weights,metadata)
    write_json(root/'history.json',{'history':history,'question_weighting':'each query contributes one mean objective',
                                  'best_epoch':best_epoch})
    return {'checkpoint':str(root),'best_epoch':best_epoch,'train_queries':len(train),'validation_queries':len(val)}
