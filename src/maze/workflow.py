"""Freeze input-only requests, collect once, and grade in a separate stage."""
from __future__ import annotations
from dataclasses import asdict
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import math
import random
import re
import time
from typing import Protocol
from maze.data import Query, Exemplar, render_messages
from maze.io import identity, read_json, read_jsonl, write_json, write_jsonl, append_durable
from maze.planning import Candidate, allocate, sample_paths, vote
from maze.backends.chat_api import Generation


class Generator(Protocol):
    def generate(self, request:dict) -> Generation: ...


def _decoding(value:dict) -> dict:
    if not {'temperature','max_tokens'}<=set(value) or not set(value)<={'temperature','top_p','max_tokens','stop','chat_template_kwargs'}:
        raise ValueError('Use explicit supported decoding settings')
    if isinstance(value['temperature'],bool) or not isinstance(value['temperature'],(float,int)) or not math.isfinite(value['temperature']) or not 0<=value['temperature']<=2:
        raise ValueError('temperature must be finite in [0,2]')
    if type(value['max_tokens']) is not int or value['max_tokens']<1:
        raise ValueError('Positive max_tokens required')
    if 'top_p' in value and (isinstance(value['top_p'],bool) or not isinstance(value['top_p'],(float,int)) or not 0<value['top_p']<=1):
        raise ValueError('top_p must be in (0,1]')
    if 'stop' in value and (not isinstance(value['stop'],list) or any(not isinstance(x,str) or not x for x in value['stop'])):
        raise ValueError('stop must be a list of nonempty strings')
    if 'chat_template_kwargs' in value and (not isinstance(value['chat_template_kwargs'],dict) or set(value['chat_template_kwargs'])!={'enable_thinking'} or type(value['chat_template_kwargs']['enable_thinking']) is not bool):
        raise ValueError('Only explicit boolean enable_thinking is supported in template settings')
    return dict(value)


def prepare(bank:dict[str,Exemplar], queries:dict[str,Query], system:str, config:dict,
            *, features=None, scorer=None, checkpoint_metadata:dict|None=None,
            anchor_paths:dict|None=None, paths_by_query:dict|None=None) -> dict:
    """The function has no gold-answer, correctness or response-length argument."""
    required={'model','model_revision','policy','calls','shots','pool_size','seed','decoding','tie_rule'}
    if not required<=set(config) or not set(config)<=required|{'repetitions','selection','purpose'}:
        raise ValueError('Unexpected or missing planning setting')
    if str(config.get('model_revision','')).startswith('SET_'):
        raise ValueError('Set model_revision to the exact frozen serving revision before preparing a run')
    if not isinstance(config['model'],str) or not config['model'].strip() or not isinstance(config['model_revision'],str) or not config['model_revision'].strip():
        raise ValueError('Record the target model and serving model revision')
    decoding=_decoding(config['decoding'])
    if config['tie_rule'] not in {'first','seeded'}:
        raise ValueError('Freeze a vote tie rule before collection')
    if config['policy'] not in {'repeat','topk','independent','allocation'}:
        raise ValueError('Unknown allocation policy')
    selection=config.get('selection','energy')
    if selection not in {'energy','uniform','retrieval'}:
        raise ValueError('Selection must be energy, uniform or retrieval')
    if selection=='energy' and (features is None or scorer is None):
        raise ValueError('Energy planning requires cached features and a fitted scorer')
    if selection=='retrieval' and features is None:
        raise ValueError('Retrieval planning requires cached features')
    if selection=='retrieval' and (config['policy']!='repeat' or config['pool_size']!=1):
        raise ValueError('Retrieval reference repeats one deterministic retrieved path (pool_size=1)')
    if not queries or not bank:
        raise ValueError('Nonempty bank and queries required')
    purpose=config.get('purpose','evaluation')
    if purpose not in {'evaluation','offline_training'}:
        raise ValueError('purpose must be evaluation or offline_training')
    if purpose=='evaluation' and {e.key for e in bank.values()} & {q.key for q in queries.values()}:
        raise ValueError('Evaluation queries must not appear in the exemplar bank')
    if features is not None:
        for row in list(bank.values())+list(queries.values()):
            if row.id not in features.lookup or features.nodes[features.lookup[row.id]]['question_key']!=row.key:
                raise ValueError('Feature-store question identity does not match input records')
    if selection=='energy':
        if config['shots']!=scorer.config.shots or features.states.shape[1:]!=(scorer.config.views,scorer.config.dim):
            raise ValueError('Scorer dimensions or path length do not match planned inputs')
        if features.recipe['start_mode']!=scorer.config.start_mode:
            raise ValueError('Scorer and feature start modes disagree')
    metadata=checkpoint_metadata or {}
    used=set(metadata.get('train_question_keys',[]))|set(metadata.get('validation_question_keys',[]))
    if used&{q.key for q in queries.values()}:
        raise ValueError('Evaluation questions overlap checkpoint fitting/model selection')
    projected=scorer.project(features.states) if selection=='energy' else None
    requests=[]; plans=[]
    for qi,(qid,query) in enumerate(sorted(queries.items())):
        eligible=[e.id for e in bank.values() if e.key!=query.key]
        seed=config['seed']+qi*1000003
        count=config['calls'] if config['policy']=='independent' else 1
        pools=[]
        for p in range(count):
            if paths_by_query is not None:
                if config['policy']=='independent' or selection=='retrieval':
                    raise ValueError('Explicit candidate paths use one shared non-retrieval pool')
                if set(paths_by_query)!=set(queries):raise ValueError('Candidate/query membership mismatch')
                paths=[tuple(path) for path in paths_by_query[qid]]
                if len(paths)!=config['pool_size'] or len(set(paths))!=len(paths):
                    raise ValueError('Explicit path count must equal pool_size, with no duplicates')
                if any(len(path)!=config['shots'] or len(set(path))!=len(path) or any(e not in eligible for e in path) for path in paths):
                    raise ValueError('Explicit paths must contain distinct eligible exemplars')
            elif selection=='retrieval':
                import numpy as np
                state=features.states[features.lookup[qid],-1].astype(float)
                denom=max(float(np.linalg.norm(state)),1e-8)
                def similarity(eid):
                    node=features.states[features.lookup[eid],-1].astype(float)
                    return float(node@state)/max(float(np.linalg.norm(node))*denom,1e-8)
                ranked=sorted(eligible,key=lambda eid:(-similarity(eid),eid))
                if len(ranked)<config['shots']:raise ValueError('Bank too small')
                paths=[tuple(ranked[:config['shots']])]
            else:
                paths=sample_paths(eligible,config['shots'],config['pool_size'],seed+p)
            anchor=tuple(anchor_paths[qid]) if anchor_paths and qid in anchor_paths else None
            if anchor and anchor not in paths:
                if any(e not in eligible for e in anchor):raise ValueError('Invalid retrieval anchor')
                paths[-1]=anchor
            if selection=='energy':
                scores=scorer.score_projected(projected,features.path_indices(qid,paths))
            elif selection=='uniform':
                rng=random.Random(seed+p+17);scores=[rng.random() for _ in paths]
            else:scores=[0.]
            pools.append([Candidate(tuple(path),float(score)) for path,score in zip(paths,scores)])
        slots=allocate(pools,config['policy'],config['calls'],seed=seed,anchor=anchor,
                       repetitions=config.get('repetitions'))
        plans.append({'query_id':qid,'scored_candidates':sum(map(len,pools)),
                      'candidates':[[{'path':list(c.path),'score':c.score} for c in pool] for pool in pools]})
        for slot in slots:
            request={'query_id':qid,'question_key':query.key,'slot':slot.slot,'pool':slot.pool,
                'path':list(slot.path),'seed':slot.seed,'model':config['model'],
                'model_revision':config['model_revision'],'decoding':decoding,
                'messages':render_messages(system,bank,slot.path,query)}
            request['request_id']=identity(request);requests.append(request)
    body={'schema':1,'config':config,'requests':requests,'planning':plans,
          'query_encoder_tokens':{q:features.recipe.get('node_token_counts',{}).get(q) if features else None for q in queries},
          'planning_target_calls':0,'role':'prospective_input_only_plan'}
    body['plan_id']=identity(body)
    return body


def validate_plan(plan:dict) -> None:
    body={k:v for k,v in plan.items() if k!='plan_id'}
    if plan.get('schema')!=1 or plan.get('plan_id')!=identity(body):
        raise ValueError('Plan identity mismatch')
    requests=plan['requests'];ids=set()
    for request in requests:
        value={k:v for k,v in request.items() if k!='request_id'}
        if request['request_id']!=identity(value) or request['request_id'] in ids:
            raise ValueError('Request identity mismatch or duplicate')
        ids.add(request['request_id']);_decoding(request['decoding'])
    if not requests:
        raise ValueError('Empty request plan')


def collect(plan:dict, target:Generator, directory:str|Path) -> dict:
    """One physical attempt per request ID, with durable pre-call claims.

    An interrupted/failed attempt is never silently retried on resume. Preserve
    the directory and investigate the serving log before designing a new run.
    """
    validate_plan(plan)
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    lock=root/'.collector.lock'
    # Exclusive creation prevents concurrent collectors sharing this output directory.
    with lock.open('x') as lockfile:lockfile.write(plan['plan_id'])
    try:
        saved=root/'plan.json'
        if saved.exists():
            if read_json(saved)!=plan:raise ValueError('This directory belongs to a different plan')
        else:write_json(saved,plan)
        events_path=root/'events.jsonl'
        events=read_jsonl(events_path) if events_path.exists() else []
        expected={r['request_id'] for r in plan['requests']}
        claims=set();complete={}
        for event in events:
            rid=event.get('request_id')
            if rid not in expected:raise ValueError('Unknown request in event log')
            if event['event']=='claimed':
                if rid in claims:raise ValueError('Duplicate physical request claim')
                claims.add(rid)
            elif event['event']=='completed':
                if rid not in claims or rid in complete:raise ValueError('Unmatched or duplicate completion')
                complete[rid]=event
            elif event['event']!='unresolved':raise ValueError('Unknown event type')
        unresolved=claims-set(complete)
        if unresolved:
            raise RuntimeError(f'{len(unresolved)} physical requests have unresolved outcomes; automatic retry is disabled')
        for request in plan['requests']:
            rid=request['request_id']
            if rid in complete:continue
            append_durable(events_path,{'event':'claimed','request_id':rid,'time':time.time()})
            start=time.perf_counter()
            try:
                generation=target.generate(request)
                if not isinstance(generation,Generation):raise TypeError('Target must return Generation')
            except Exception as exc:
                # Do not write a provider exception string; it can contain credentials or prompt text.
                append_durable(events_path,{'event':'unresolved','request_id':rid,'exception_type':type(exc).__name__})
                raise RuntimeError('Physical request unresolved; recorded without automatic retry') from exc
            record={'event':'completed','request_id':rid,'generation':asdict(generation),
                    'wall_seconds':time.perf_counter()-start}
            append_durable(events_path,record);complete[rid]=record
        summary={'plan_id':plan['plan_id'],'completed_requests':len(complete),'automatic_retries':0,
                 'all_complete':len(complete)==len(plan['requests'])}
        write_json(root/'collection.json',summary,overwrite=True)
        return summary
    finally:
        lock.unlink(missing_ok=True)


def extract_answer(text:str,kind:str) -> str|None:
    if not isinstance(text,str):raise ValueError('Answer parser expects text')
    matches=re.findall(r'(?im)^\s*(?:final\s+answer|answer)\s*:\s*(.+?)\s*$',text)
    if matches:value=matches[-1].strip()
    else:
        boxes=re.findall(r'\\boxed\{([^{}]+)\}',text)
        value=boxes[-1].strip() if boxes else text.strip()
    if kind=='choice':
        match=re.fullmatch(r'\(?([A-Z])\)?[.\s]*',value)
        return match.group(1) if match else None
    if kind=='numeric':
        try:
            if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?',value.replace(',','')):return None
            return str(Fraction(value.replace(',','')))
        except (ValueError,ZeroDivisionError):return None
    if kind=='exact':return value if value else None
    raise ValueError('Supported graders: choice, numeric, exact. Symbolic tasks require a benchmark-specific evaluator.')


def grade(plan:dict, events:list[dict], answers:list[dict], *, kind:str) -> dict:
    validate_plan(plan)
    gold={}
    for row in answers:
        if set(row)!={'query_id','answer'} or row['query_id'] in gold or not isinstance(row['answer'],str):
            raise ValueError('Strict unique query_id/answer records required')
        answer=extract_answer(row['answer'],kind)
        if answer is None:raise ValueError('Gold answer is not valid for the chosen evaluator')
        gold[row['query_id']]=answer
    outputs={}
    for event in events:
        if event['event']=='completed':
            if event['request_id'] in outputs:raise ValueError('Duplicate response ID')
            Generation(**event['generation'])
            outputs[event['request_id']]=event['generation']
    if set(outputs)!={r['request_id'] for r in plan['requests']}:
        raise ValueError('Every planned request needs one completed response; partial runs are not silently scored')
    groups=defaultdict(list)
    for r in plan['requests']:groups[r['query_id']].append((r,outputs[r['request_id']]))
    if set(gold)!=set(groups):raise ValueError('Gold-answer membership must match the frozen query cohort exactly')
    results=[]
    for qid,rows in sorted(groups.items()):
        rows.sort(key=lambda x:x[0]['slot'])
        parsed=[extract_answer(g['text'],kind) for _,g in rows]
        answer=vote(parsed,seed=rows[0][0]['seed'],tie=plan['config']['tie_rule'])
        total_in=None if any(g['input_tokens'] is None for _,g in rows) else sum(g['input_tokens'] for _,g in rows)
        total_out=None if any(g['output_tokens'] is None for _,g in rows) else sum(g['output_tokens'] for _,g in rows)
        encoder=plan['query_encoder_tokens'][qid]
        known=all(x is not None for x in (total_in,total_out,encoder))
        results.append({'query_id':qid,'answer':answer,'correct':int(answer==gold[qid]),'calls':len(rows),
            'invalid_calls':sum(x is None for x in parsed),'truncated_calls':sum(g['finish_reason']=='length' for _,g in rows),
            'input_tokens':total_in,'output_tokens':total_out,'query_encoder_tokens':encoder,
            'processed_tokens':total_in+total_out+encoder if known else None})
    return {'schema':1,'plan_id':plan['plan_id'],'questions':len(results),
            'accuracy_exact':str(Fraction(sum(x['correct'] for x in results),len(results))),
            'grader':kind,'role':'fresh_run_evaluation','outcomes':results}
