#!/usr/bin/env python3
"""Propose a fixed input-only chain of local edits for offline supervision."""
from pathlib import Path
import argparse
import math
import random
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from maze.data import load_bank,load_queries
from maze.io import write_json


def propose(ids,shots,count,seed):
    if not 1<=shots<=len(ids) or not 1<=count<=math.perm(len(ids),shots):raise ValueError('Infeasible shot/path count')
    rng=random.Random(seed);current=tuple(rng.sample(sorted(ids),shots));paths=[current];seen={current}
    attempts=0
    while len(paths)<count:
        attempts+=1
        if attempts>count*1000:raise ValueError('Unable to construct the requested distinct edit chain')
        path=list(current)
        alternatives=sorted(set(ids)-set(path))
        if alternatives and (shots==1 or rng.random()<.5):path[rng.randrange(shots)]=rng.choice(alternatives)
        elif shots>=2:
            i,j=rng.sample(range(shots),2);path[i],path[j]=path[j],path[i]
        else:raise ValueError('No eligible edits')
        trial=tuple(path)
        if trial in seen:continue
        seen.add(trial);paths.append(trial);current=trial
    return paths


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True);p.add_argument('--queries',type=Path,required=True)
    p.add_argument('--shots',type=int,default=4);p.add_argument('--count',type=int,default=32)
    p.add_argument('--seed',type=int,default=20261020);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();bank=load_bank(a.bank);queries=load_queries(a.queries)
    paths={q.id:propose([e.id for e in bank.values() if e.key!=q.key],a.shots,a.count,a.seed+i)
           for i,q in enumerate(sorted(queries.values(),key=lambda q:q.id))}
    write_json(a.output,paths)
    print(f'Prepared {len(paths)} input-only pools. No target calls or labels were used.')


if __name__=='__main__':main()
