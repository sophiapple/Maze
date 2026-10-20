#!/usr/bin/env python3
"""Recompute the published result tables using only the Python standard library."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from maze.results import reconstruct, table, write_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/reproduced')
    args=parser.parse_args()
    report=reconstruct(ROOT,verify=True)
    write_report(report,args.output)
    print(table(report))
    print(f"Verified {report['records']} saved outcomes; {report['comparison_count']} paired comparisons.")
    print('Saved-outcome reconstruction, not new target generation or independent regrading.')
    print(f'Outputs: {args.output}')


if __name__=='__main__':
    main()
