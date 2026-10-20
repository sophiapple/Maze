#!/usr/bin/env python3
"""Fit a residual utility head with disjoint fitting/evaluation query identities."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from maze.calibration import fit_residual
from maze.io import read_json,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--alpha',type=float,default=10.)
    a=p.parse_args();v=read_json(a.input)
    if set(v)!={'features','utility_differences','query_keys','held_out_keys'}:
        raise ValueError('Calibration needs input-side features, observed differences and disjoint question IDs')
    model=fit_residual(v['features'],v['utility_differences'],v['query_keys'],alpha=a.alpha,held_out_keys=v['held_out_keys'])
    write_json(a.output,model);print(f'Calibrated {len(model["fit_question_keys"])} fitting queries.')


if __name__=='__main__':main()
