#!/usr/bin/env python3
"""Convert locally obtained benchmark records to Maze inputs and separate labels.

No benchmark is downloaded or redistributed. Use source-native splits under
its access terms. This converter defines a fresh protocol, not the archived cohorts.
"""
from pathlib import Path
import argparse
import hashlib
import random
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from maze.data import question_key
from maze.io import read_jsonl,write_jsonl,write_json,sha256


def canonical(row,benchmark,seed):
    if benchmark=='mmlu_pro':
        q=row['question'];choices=row['options'];idx=row.get('answer_index')
        answer=chr(65+idx) if type(idx) is int else row['answer']
        rationale=row.get('cot_content','')
    elif benchmark=='gpqa':
        q=row['Question'];options=[row['Correct Answer']]+[row[f'Incorrect Answer {i}'] for i in (1,2,3)]
        order=list(range(4));random.Random(seed).shuffle(order)
        choices=[options[i] for i in order];answer=chr(65+order.index(0));rationale=''
    elif benchmark=='gsm8k':
        q=row['question'];solution=row['answer']
        if '####' not in solution:raise ValueError('GSM8K solution must contain its final-answer delimiter')
        rationale,answer=solution.rsplit('####',1);choices=[]
    elif benchmark in {'aime','math500'}:
        q=row.get('problem',row.get('Problem'));answer=str(row.get('answer',row.get('Answer','')))
        if q is None or not answer.strip():raise ValueError('AIME/MATH records need problem and answer')
        rationale=row.get('solution','');choices=[]
    elif benchmark=='canonical':
        q=row['question'];answer=str(row['answer']);choices=row.get('choices',[]);rationale=row.get('rationale','')
    else:raise ValueError('Unknown benchmark')
    if not isinstance(answer,str) or not answer.strip():raise ValueError('Empty answer')
    key=question_key(q)
    return key,{'question':q,'choices':choices,'answer':answer.strip(),'rationale':rationale.strip()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--benchmark',choices=['mmlu_pro','gpqa','gsm8k','aime','math500','canonical'],required=True)
    p.add_argument('--split',choices=['train','validation','test'],required=True)
    p.add_argument('--seed',type=int,default=20261020)
    a=p.parse_args();seen=set();queries=[];answers=[];bank=[]
    for i,row in enumerate(read_jsonl(a.input)):
        key,record=canonical(row,a.benchmark,a.seed+i)
        if key in seen:raise ValueError('Duplicate normalized question in source split')
        seen.add(key);qid='q_'+key[:24]
        queries.append({'id':qid,'question':record['question'],'choices':record['choices']})
        answers.append({'query_id':qid,'answer':record['answer']})
        if a.split=='train':bank.append({'id':'e_'+key[:24],**record})
    a.output.mkdir(parents=True,exist_ok=False)
    write_jsonl(a.output/'queries.jsonl',queries);write_jsonl(a.output/'answers.jsonl',answers)
    if bank:write_jsonl(a.output/'bank.jsonl',bank)
    write_json(a.output/'manifest.json',{'benchmark':a.benchmark,'split':a.split,'source_sha256':sha256(a.input),
        'questions':len(queries),'question_keys':sorted(seen),'choice_shuffle_seed':a.seed,
        'protocol':'fresh_local_dataset_conversion'})
    print(f'Prepared {len(queries)} {a.split} queries with labels in a separate file.')


if __name__=='__main__':main()
