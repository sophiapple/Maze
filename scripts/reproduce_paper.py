#!/usr/bin/env python3
"""Verify camera-ready table transcription and recompute reported contrasts."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from maze_paper.reported import reconstruct, table, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/paper')
    parser.add_argument('--paper', type=Path, help='Optional original October 20 main.tex for full-source hash verification')
    args = parser.parse_args()
    report = reconstruct(ROOT, args.paper)
    write_report(report, args.output)
    print(table(report))


if __name__ == '__main__':
    main()
