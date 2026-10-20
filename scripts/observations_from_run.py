#!/usr/bin/env python3
"""Convert completed offline requests into query/path correctness targets."""
from collections import defaultdict
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from maze.io import read_json,read_jsonl,write_jsonl
from maze.workflow import validate_plan,extract_answer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','events','answers','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--split',choices=['train','validation'],required=True)
    p.add_argument('--kind',choices=['choice','numeric','exact'],required=True)
    a=p.parse_args();plan=read_json(a.plan);validate_plan(plan)
    truth=read_jsonl(a.answers)
    if any(set(r)!={'query_id','answer'} for r in truth) or len({r['query_id'] for r in truth})!=len(truth):
        raise ValueError('Unique query_id/answer evaluation records required')
    gold={r['query_id']:extract_answer(r['answer'],a.kind) for r in truth}
    if None in gold.values() or set(gold)!={r['query_id'] for r in plan['requests']}:
        raise ValueError('Gold membership or answer format mismatch')
    outputs={}
    for e in read_jsonl(a.events):
        if e['event']=='completed':
            if e['request_id'] in outputs:raise ValueError('Duplicate response')
            outputs[e['request_id']]=e['generation']
    if set(outputs)!={r['request_id'] for r in plan['requests']}:raise ValueError('Every planned request must be present')
    groups=defaultdict(list);first={}
    for r in plan['requests']:
        q=r['query_id'];path=tuple(r['path']);first.setdefault(q,path)
        parsed=extract_answer(outputs[r['request_id']]['text'],a.kind)
        groups[q,path].append(int(parsed==gold[q]))
    rows=[{'query_id':q,'path':list(path),'target':sum(values)/len(values),'split':a.split,'reference':path==first[q]}
          for (q,path),values in sorted(groups.items())]
    write_jsonl(a.output,rows)
    print(f'Wrote {len(rows)} {a.split} query/path targets. The first frozen request path is the reference.')


if __name__=='__main__':main()
