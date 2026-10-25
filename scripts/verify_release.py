#!/usr/bin/env python3
"""Check deterministic reports, README evidence, and repository integrity offline."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from maze.results import reconstruct, table, write_report
from maze_paper.reported import reconstruct as paper_report, write_report as write_paper


def compare_directory(actual: Path, expected: Path):
    names = {p.name for p in actual.iterdir() if p.is_file()}
    if names != {p.name for p in expected.iterdir() if p.is_file()}:
        raise ValueError('Report file membership differs: '+str(expected))
    for name in sorted(names):
        if (actual/name).read_bytes() != (expected/name).read_bytes():
            raise ValueError('Stale or modified generated report: '+str(expected/name))


def main():
    report = reconstruct(ROOT)
    paper = paper_report(ROOT)
    with tempfile.TemporaryDirectory(prefix='maze-verify-') as tmp:
        tmp = Path(tmp)
        write_report(report, tmp/'development')
        write_paper(paper, tmp/'paper')
        compare_directory(tmp/'development', ROOT/'results/reproduced')
        compare_directory(tmp/'paper', ROOT/'results/paper')
    readme = (ROOT/'README.md').read_text()
    begin, end = '<!-- BEGIN GENERATED DEVELOPMENT HIGHLIGHTS -->', '<!-- END GENERATED DEVELOPMENT HIGHLIGHTS -->'
    if readme.count(begin)!=1 or readme.count(end)!=1:
        raise ValueError('Missing or duplicate README evidence markers')
    if readme.split(begin)[1].split(end)[0].strip() != table(report, highlights=True).strip():
        raise ValueError('README highlights differ from reconstructed evidence')
    subprocess.run([sys.executable, str(ROOT/'scripts/audit_repository.py')], check=True)
    print(json.dumps({'state':'passed', 'outcomes':report['records'],
        'arms':report['arm_count'], 'comparisons':report['comparison_count'],
        'paper_tables':paper['table_count'], 'paper_main_values':len(paper['main_table_values']),
        'gpqa32_vote_audit':report['gpqa32_vote_audit'], 'network_required':False}, indent=2))


if __name__ == '__main__':
    main()
