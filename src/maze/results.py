"""Exact result reconstruction from published question-level numeric records.

This module reads saved correctness indicators. It does not recreate prompts,
regrade text, train a scorer, or execute a target model.
"""
from __future__ import annotations
import csv
from fractions import Fraction
import io
from pathlib import Path
import random
from maze.io import read_json, read_jsonl, sha256, write_json
from maze_paper.evidence import aggregate, verify_reference, verify_vote_records


def _quantile(values: list[float], q: float) -> float:
    xs = sorted(values)
    x = q * (len(xs)-1)
    lo = int(x)
    hi = min(lo+1, len(xs)-1)
    return xs[lo] + (x-lo)*(xs[hi]-xs[lo])


def paired_interval(differences: list[int], seed: int, replicates: int) -> list[float]:
    if not differences or any(type(x) is not int or x not in (-1,0,1) for x in differences):
        raise ValueError("Expected aligned binary correctness differences")
    if type(replicates) is not int or replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    rng = random.Random(seed)
    n = len(differences)
    means = [sum(differences[rng.randrange(n)] for _ in range(n)) / n
             for _ in range(replicates)]
    return [100*_quantile(means, q) for q in (.025, .975)]


def reconstruct(root: str | Path, *, verify: bool = True) -> dict:
    root = Path(root)
    ledger = root/'evidence/development/outcomes.jsonl'
    rows = read_jsonl(ledger)
    reference = read_json(root/'evidence/development/reference.json')
    config = read_json(root/'configs/results.json')
    if verify:
        provenance = read_json(root/'provenance/inputs.json')
        if sha256(ledger) != provenance['outcome_ledger_sha256']:
            raise ValueError("Published outcome ledger digest mismatch")
        for name, expected in provenance['additional_evidence_sha256'].items():
            if sha256(root/name) != expected:
                raise ValueError('Additional evidence digest mismatch: '+name)
    values = aggregate(rows)
    if set(values) != {c['id'] for c in reference['cohorts']}:
        raise ValueError("Missing or unexpected cohort")
    verify_reference(values, reference)
    vote_audit = verify_vote_records(read_jsonl(root/'evidence/development/gpqa32_vote_calls.jsonl'),
        rows, read_json(root/'evidence/development/gpqa32_vote_reference.json'))
    overlap = read_json(root/'evidence/development/cohort_overlap.json')
    by_arm = {}
    for row in rows:
        by_arm.setdefault((row['cohort'], row['arm']), {})[row['question']] = row
    pairs = overlap['pairs']
    for cohort in ('gpqa32_v166', 'gpqa99_v206'):
        ids = [p[cohort] for p in pairs]
        known = {r['question'] for r in rows if r['cohort']==cohort}
        if len(ids) != len(set(ids)) or not set(ids) <= known:
            raise ValueError('Invalid cohort overlap mapping')
    memberships = sum(len({r['question'] for r in rows if r['cohort']==c}) for c in values)
    if (memberships != overlap['cohort_question_memberships'] or
            memberships-len(pairs) != overlap['distinct_source_question_ids']):
        raise ValueError('Cohort overlap count mismatch')
    comparisons = []
    ids = set()
    for item in config['comparisons']:
        if item['id'] in ids:
            raise ValueError("Duplicate comparison")
        ids.add(item['id'])
        cohort, a, b = item['cohort'], item['maze_arm'], item['baseline_arm']
        ar, br = by_arm[cohort,a], by_arm[cohort,b]
        if set(ar) != set(br):
            raise ValueError("Paired comparison needs identical question membership")
        va, vb = values[cohort][a], values[cohort][b]
        aa, ab = Fraction(va['accuracy_exact']), Fraction(vb['accuracy_exact'])
        ta, tb = Fraction(va['processed_tokens_exact']), Fraction(vb['processed_tokens_exact'])
        differences = [ar[q]['correct'] - br[q]['correct'] for q in sorted(ar)]
        comparisons.append(dict(item, questions=len(ar), maze_correct=va['correct'],
            baseline_correct=vb['correct'], maze_accuracy_pct=100*float(aa),
            baseline_accuracy_pct=100*float(ab), gain_pp=100*float(aa-ab),
            gain_exact=str(aa-ab), maze_calls=va['calls_exact'], baseline_calls=vb['calls_exact'],
            token_ratio=float(ta/tb), token_ratio_exact=str(ta/tb),
            maze_mean_tokens_exact=str(ta), baseline_mean_tokens_exact=str(tb),
            paired_wins=differences.count(1), paired_losses=differences.count(-1),
            paired_ties=differences.count(0),
            gain_ci95_pp=paired_interval(differences, config['bootstrap']['seed'],
                                        config['bootstrap']['replicates'])))
    return {'schema':1, 'state':'verified_saved_outcome_reconstruction',
        'records':len(rows), 'unique_questions':memberships-len(pairs),
        'cohort_question_memberships':memberships, 'shared_gpqa_question_ids':len(pairs),
        'cohort_count':len(values),'arm_count':sum(map(len, values.values())),
        'comparison_count':len(comparisons),'target':config['target'],'thinking':config['thinking'],
        'shots':config['shots'],'bootstrap':config['bootstrap'],
        'scope':reference['scope'],'fresh_generation':False,'independent_regrading':False,
        'trained_checkpoint_included_for_recorded_experiments':False,
        'outcomes_sha256':sha256(ledger),'cohorts':values, 'comparisons':comparisons,
        'cohort_displays':config['cohort_displays'], 'gpqa32_vote_audit':vote_audit,
        'cost_scope':'Per-policy deployment target input + output + recorded query encoder. Cached requests overlap between policies; arm costs are not additive physical research usage. Offline bank/setup/training and serving latency are separate.',
        'offline_bank_tokens':{c['id']:c['offline_bank_tokens'] for c in reference['cohorts']},
        'complete_project_cost':None,'end_to_end_latency':None}


def table(report: dict, *, highlights: bool = False) -> str:
    lines=['| Cohort | Maze policy | Baseline | Maze accuracy | Baseline accuracy | Gain | Calls | Token ratio |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for r in report['comparisons']:
        if highlights and not r['highlight']:
            continue
        name=report['cohort_displays'][r['cohort']]
        accuracy=f"{r['maze_accuracy_pct']:.2f}%"
        if r['gain_pp']>0:
            accuracy='**'+accuracy+'**'
        lines.append(f"| {name} | {r['maze_label']} | {r['baseline_label']} | "
                     f"{accuracy} | {r['baseline_accuracy_pct']:.2f}% | "
                     f"{r['gain_pp']:+.2f} pp | {r['maze_calls']} / {r['baseline_calls']} | {r['token_ratio']:.3f}× |")
    return '\n'.join(lines)+'\n'


def write_report(report: dict, output: str | Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'summary.json', report)
    (output/'comparisons.md').write_text(table(report),encoding='utf-8')
    (output/'highlights.md').write_text(table(report,highlights=True),encoding='utf-8')
    arms = []
    for cohort, values in report['cohorts'].items():
        for arm, value in values.items():
            arms.append(dict(cohort=cohort, arm=arm, **value))
    with (output/'all_arms.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(arms[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(arms)
    names=['id','cohort','questions','maze_label','baseline_label','maze_correct','baseline_correct',
           'maze_accuracy_pct','baseline_accuracy_pct','gain_pp','gain_exact','maze_calls','baseline_calls',
           'token_ratio','token_ratio_exact','maze_mean_tokens_exact','baseline_mean_tokens_exact',
           'paired_wins','paired_losses','paired_ties','gain_ci95_pp']
    with (output/'comparisons.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=names,extrasaction='ignore',lineterminator='\n')
        writer.writeheader(); writer.writerows(report['comparisons'])
