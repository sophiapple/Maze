"""Command-line workflows for Maze."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from maze.io import read_json,read_jsonl,write_json,sha256


def main(argv=None):
    p=argparse.ArgumentParser(prog='maze',description='Prompt-trajectory planning for efficient reasoning.')
    commands=p.add_subparsers(dest='command',required=True)
    results=commands.add_parser('results',help='Reconstruct included recorded results (no model execution)')
    results.add_argument('--root',type=Path,default=Path.cwd())
    results.add_argument('--output',type=Path)
    results.add_argument('--verify',action='store_true',help='Explicit verification flag; verification is always enabled')
    for name in ('demo','train-demo'):
        cmd=commands.add_parser(name,help='Synthetic software integration test, not a benchmark')
        cmd.add_argument('--output',type=Path,required=True)
    train=commands.add_parser('train',help='Fit a planner using frozen features and offline correctness records')
    for x in ('features','observations','config','output'):train.add_argument('--'+x,required=True,type=Path)
    encode=commands.add_parser('encode',help='Compute frozen encoder states using optional Transformers backend')
    for x in ('bank','queries','system','output'):encode.add_argument('--'+x,required=True,type=Path)
    encode.add_argument('--model',required=True);encode.add_argument('--revision',required=True)
    encode.add_argument('--layers',required=True,help='Comma-separated hidden-state tuple indices, including 0 for embeddings')
    encode.add_argument('--device',default='cpu');encode.add_argument('--dtype',default='float32')
    encode.add_argument('--max-tokens',type=int,default=8192)
    encode.add_argument('--start-mode',choices=['encoded_system_full_qa','zero_start_question_nodes'],default='encoded_system_full_qa')
    plan=commands.add_parser('prepare',help='Create input-only frozen requests, without generating answers')
    for x in ('bank','queries','system','config','output'):plan.add_argument('--'+x,required=True,type=Path)
    plan.add_argument('--features',type=Path)
    plan.add_argument('--candidates',type=Path,help='Optional query-ID to ordered candidate-path arrays')
    plan.add_argument('--checkpoint',type=Path,action='append',help='Repeat this flag for equal-weight scorer consensus')
    collect=commands.add_parser('collect',help='Execute a frozen request plan on a configured chat endpoint')
    collect.add_argument('--plan',required=True,type=Path);collect.add_argument('--output',required=True,type=Path)
    collect.add_argument('--base-url',default=os.environ.get('MAZE_API_BASE_URL'))
    collect.add_argument('--key-env',default='MAZE_API_KEY');collect.add_argument('--timeout',type=float,default=180)
    collect.add_argument('--allow-api-calls',action='store_true',help='Required acknowledgement that model calls may incur cost')
    grade=commands.add_parser('grade',help='Evaluate completed responses in a separate label-aware stage')
    for x in ('plan','events','answers','output'):grade.add_argument('--'+x,required=True,type=Path)
    grade.add_argument('--kind',choices=['choice','numeric','exact'],required=True)
    args=p.parse_args(argv)
    try:
        if args.command=='results':
            from maze.results import reconstruct,table,write_report
            report=reconstruct(args.root)
            if args.output:write_report(report,args.output)
            print(table(report))
            print('Verified saved-outcome reconstruction; no model generation or independent text regrading.')
            return 0
        if args.command in {'demo','train-demo'}:
            from maze.demo import run_demo,train_demo
            value=(run_demo if args.command=='demo' else train_demo)(args.output)
        elif args.command=='train':
            from maze.features import FeatureStore
            from maze.scorers.numpy_energy import EnergyConfig
            from maze.training import fit
            cfg=read_json(args.config)
            value=fit(FeatureStore.load(args.features),read_jsonl(args.observations),EnergyConfig(**cfg['architecture']),cfg['training'],args.output)
        elif args.command=='encode':
            from maze.backends.transformers_encoder import encode
            encode(str(args.bank),str(args.queries),str(args.system),str(args.output),model_name=args.model,
                   revision=args.revision,layers=[int(x) for x in args.layers.split(',')],device=args.device,
                   dtype=args.dtype,max_tokens=args.max_tokens,start_mode=args.start_mode)
            value={'features':str(args.output)}
        elif args.command=='prepare':
            from maze.data import load_bank,load_queries
            from maze.features import FeatureStore
            from maze.scorers.numpy_energy import load_checkpoint
            from maze.scorers.ensemble import EnergyEnsemble
            from maze.workflow import prepare
            store=FeatureStore.load(args.features) if args.features else None
            if store:
                for key,path in [('bank_sha256',args.bank),('queries_sha256',args.queries),('system_sha256',args.system)]:
                    if key in store.recipe and store.recipe[key]!=sha256(path):raise ValueError('Feature source mismatch: '+key)
            models=[load_checkpoint(path) for path in (args.checkpoint or [])]
            scorer=(models[0][0] if len(models)==1 else EnergyEnsemble([v[0] for v in models])) if models else None
            metadata={'train_question_keys':sorted({k for _,info in models for k in info['metadata'].get('train_question_keys',[])}),
                      'validation_question_keys':sorted({k for _,info in models for k in info['metadata'].get('validation_question_keys',[])})}
            if models and store:
                for _,info in models:
                    recipe=info['metadata'].get('feature_recipe',{})
                    for field in ('model','revision','layers','pooling','start_mode','node_encoding'):
                        if field in recipe and recipe[field]!=store.recipe.get(field):
                            raise ValueError('Checkpoint/encoder recipe mismatch: '+field)
            value=prepare(load_bank(args.bank),load_queries(args.queries),args.system.read_text(),read_json(args.config),
                          features=store,scorer=scorer,checkpoint_metadata=metadata,
                          paths_by_query=read_json(args.candidates) if args.candidates else None)
            write_json(args.output,value)
            value={'plan':str(args.output),'requests':len(value['requests']),'target_calls_during_planning':0}
        elif args.command=='collect':
            if not args.allow_api_calls or not args.base_url:
                raise ValueError('Collection requires --allow-api-calls and an explicit API base URL')
            from maze.backends.chat_api import ChatAPI
            from maze.workflow import collect
            value=collect(read_json(args.plan),ChatAPI(args.base_url,key_env=args.key_env,timeout=args.timeout),args.output)
        elif args.command=='grade':
            from maze.workflow import grade
            value=grade(read_json(args.plan),read_jsonl(args.events),read_jsonl(args.answers),kind=args.kind)
            write_json(args.output,value)
            value={k:v for k,v in value.items() if k!='outcomes'}
        print(json.dumps(value,indent=2))
        return 0
    except (ValueError,KeyError,FileNotFoundError,FileExistsError,ImportError,RuntimeError) as exc:
        p.exit(2,f'maze: {type(exc).__name__}: {exc}\n')


if __name__=='__main__':
    main()
