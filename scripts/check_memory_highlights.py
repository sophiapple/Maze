#!/usr/bin/env python3
"""Check aggregate highlight arithmetic; this does not reproduce an experiment."""
from pathlib import Path
import json
import math
import re
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
BEGIN = '<!-- BEGIN COMPACT MEMORY HIGHLIGHTS -->'
END = '<!-- END COMPACT MEMORY HIGHLIGHTS -->'


def table(data):
    if data['schema'] != 1 or not data['rows']:
        raise ValueError('Unsupported or empty highlight record')
    seen = set()
    lines = [
        '| Panel | Maze variant | Correct / questions | Cached control | Control correct / questions | Fewer simulated target tokens |',
        '|---|---|---:|---|---:|---:|',
    ]
    for row in data['rows']:
        if row['id'] in seen:
            raise ValueError('Duplicate contrast: ' + row['id'])
        seen.add(row['id'])
        n = row['questions']
        counts = row['correct_by_conditional_replicate']
        tokens = row['mean_tokens_by_conditional_replicate']
        if type(n) is not int or n <= 0 or not counts or len(counts) != len(tokens):
            raise ValueError('Invalid denominator or conditional replicate pairing')
        if any(type(c) is not int or not 0 <= c <= n for c in counts):
            raise ValueError('Invalid correct count')
        baseline = row['comparator_correct']
        if type(baseline) is not int or not 0 <= baseline <= n:
            raise ValueError('Invalid comparator count')
        if any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0
               for t in tokens + [row['comparator_mean_tokens']]):
            raise ValueError('Invalid token usage')
        if any(not re.fullmatch(r'[0-9a-f]{64}', row[k])
               for k in ('source_sha256', 'interval_source_sha256')):
            raise ValueError('Invalid source hash')
        # Intervals were computed in the private study, not from these aggregates.
        interval = row['accuracy_gain_ci95_pp']
        if (len(interval) != 2 or any(type(v) not in (int, float) or not math.isfinite(v)
                                     for v in interval)
                or not -100 <= interval[0] <= interval[1] <= 100):
            raise ValueError('Invalid reported interval')
        correct = mean(counts)
        reduction = 100 * (1 - mean(tokens) / row['comparator_mean_tokens'])
        # This file is explicitly a selected-positive highlight list, not a full ledger.
        if correct <= baseline or reduction <= 0:
            raise ValueError('Record does not match the declared highlight scope')
        result = f'{correct:g}/{n}' + (' (mean)' if len(counts) > 1 else '')
        lines.append(f"| {row['cohort']} | {row['label']} | **{result}** | "
                     f"{row['comparator'].replace('raw', 'Raw')} | {baseline}/{n} | **{reduction:.2f}%** |")
    return '\n'.join(lines)


def verify(root=ROOT):
    data = json.loads((root/'evidence/highlights/compact_memory.json').read_text())
    expected = table(data)
    for rel in ('README.md', 'docs/COMPACT_MEMORY.md'):
        content = (root/rel).read_text()
        if content.count(BEGIN) != 1 or content.count(END) != 1:
            raise ValueError('Missing or duplicate highlight markers: ' + rel)
        if content.split(BEGIN)[1].split(END)[0].strip() != expected:
            raise ValueError('Highlight table differs from aggregate arithmetic: ' + rel)
    return {'state': 'passed', 'selected_positive_contrasts': len(data['rows']),
            'scope': 'aggregate arithmetic and table consistency only',
            'new_inference': False, 'independent_regrading': False,
            'question_level_reconstruction': False, 'intervals_recomputed': False}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
