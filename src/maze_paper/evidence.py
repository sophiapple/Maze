"""Recompute recorded-outcome aggregates; does not grade or generate responses."""
import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path

FIELDS = {"cohort", "question", "arm", "correct", "calls", "input_tokens",
          "output_tokens", "query_encoder_tokens", "processed_tokens",
          "invalid_calls", "truncated_calls"}


def aggregate(records):
    groups = defaultdict(list)
    identities = set()
    question_sets = {}
    for row in records:
        if set(row) != FIELDS:
            raise ValueError("unexpected or missing outcome fields")
        key = tuple(row[k] for k in ("cohort", "question", "arm"))
        if any(not isinstance(v, str) or not v for v in key) or key in identities:
            raise ValueError("invalid or duplicate identity")
        identities.add(key)
        if type(row["correct"]) is not int or row["correct"] not in (0, 1):
            raise ValueError("correct must be a recorded binary outcome")
        for k in FIELDS - {"cohort", "question", "arm", "correct"}:
            if type(row[k]) is not int or row[k] < 0:
                raise ValueError("counts must be nonnegative integers")
        if row["calls"] < 1 or max(row["invalid_calls"], row["truncated_calls"]) > row["calls"]:
            raise ValueError("invalid request counters")
        if row["processed_tokens"] != sum(row[k] for k in
                                         ("input_tokens", "output_tokens", "query_encoder_tokens")):
            raise ValueError("processed-token accounting mismatch")
        groups[(row["cohort"], row["arm"])].append(row)
    if not groups:
        raise ValueError("no outcomes")
    result = {}
    for (cohort, arm), rows in sorted(groups.items()):
        questions = {x["question"] for x in rows}
        if cohort in question_sets and question_sets[cohort] != questions:
            raise ValueError("unequal question membership across controls")
        question_sets[cohort] = questions
        n = len(rows)
        result.setdefault(cohort, {})[arm] = {
            "questions": n,
            "correct": sum(r["correct"] for r in rows),
            "accuracy_exact": str(Fraction(sum(r["correct"] for r in rows), n)),
            "processed_tokens_exact": str(Fraction(sum(r["processed_tokens"] for r in rows), n)),
            "calls_exact": str(Fraction(sum(r["calls"] for r in rows), n)),
            "invalid_calls": sum(r["invalid_calls"] for r in rows),
            "truncated_calls": sum(r["truncated_calls"] for r in rows),
        }
    return result


def verify_reference(aggregates, reference):
    expected = {c["id"]: c for c in reference["cohorts"]}
    if set(aggregates) != set(expected):
        raise ValueError("missing or unexpected cohort")
    for cohort, arms in aggregates.items():
        ref = expected[cohort]
        if set(arms) != set(ref["arms"]):
            raise ValueError("missing or unexpected comparator")
        for arm, result in arms.items():
            r = ref["arms"][arm]
            for key in ("accuracy_exact", "processed_tokens_exact"):
                if Fraction(result[key]) != Fraction(r[key]):
                    raise ValueError(f"{cohort}/{arm}: mismatch in {key}")
            if result["questions"] != ref["questions"] or Fraction(result["calls_exact"]) != r["calls"]:
                raise ValueError("sample count or call budget mismatch")
            for key in ("invalid_calls", "truncated_calls"):
                if r[key] is not None and result[key] != r[key]:
                    raise ValueError("technical failure count mismatch")
    return True


def verify_vote_records(calls, outcomes, metadata):
    """Reconstruct closed GPQA32 votes from opaque answer-equivalence classes.

    Equivalence classes preserve already-parsed equality, not raw response text.
    This checks voting and accounting; correctness remains author-recorded.
    Shared call IDs are retained and counted once in physical-cache accounting.
    """
    fields = {"cohort", "question", "arm", "draw", "call_id", "answer_class",
              "correct", "input_tokens", "output_tokens", "truncated"}
    groups, physical = defaultdict(list), {}
    for row in calls:
        if set(row) != fields or row['cohort'] != 'gpqa32_v166':
            raise ValueError('Unexpected vote record schema or cohort')
        for key in ('question', 'arm', 'call_id'):
            if not isinstance(row[key], str) or not row[key]:
                raise ValueError('Missing vote identity')
        for key in ('draw', 'correct', 'input_tokens', 'output_tokens', 'truncated'):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError('Vote counters must be nonnegative integers')
        if row['correct'] not in (0, 1) or row['truncated'] not in (0, 1):
            raise ValueError('Invalid binary outcome')
        answer = row['answer_class']
        if answer is not None and (not isinstance(answer, str) or not answer.startswith('a') or not answer[1:].isdigit()):
            raise ValueError('Expected opaque answer class or null')
        if answer is None and row['correct']:
            raise ValueError('An unparsed response cannot be correct')
        identity = {k: row[k] for k in fields-{'arm', 'draw'}}
        if row['call_id'] in physical and physical[row['call_id']] != identity:
            raise ValueError('Conflicting reuse of cached request ID')
        physical[row['call_id']] = identity
        groups[(row['question'], row['arm'])].append(row)
    expected = {(r['question'], r['arm']): r for r in outcomes if r['cohort']=='gpqa32_v166'}
    if set(groups) != set(expected):
        raise ValueError('Missing or unexpected vote group')
    for (question, arm), group in groups.items():
        ordered = sorted(group, key=lambda r:r['draw'])
        expected_calls = metadata['arm_calls'][arm]
        if ([r['draw'] for r in ordered] != list(range(expected_calls)) or
                len({r['call_id'] for r in ordered}) != expected_calls):
            raise ValueError('Each vote needs distinct requests in the exact frozen order')
        counts, correctness = {}, {}
        for row in ordered:
            answer = row['answer_class']
            if answer is not None:
                if answer in correctness and correctness[answer] != row['correct']:
                    raise ValueError('Conflicting correctness for one answer class')
                counts[answer] = counts.get(answer, 0)+1
                correctness[answer] = row['correct']
        # Dict insertion order preserves the earliest nonnull answer on ties.
        winner = max(counts, key=counts.get) if counts else None
        reconstructed = {'cohort':'gpqa32_v166', 'question':question, 'arm':arm,
            'correct':correctness[winner] if winner is not None else 0,
            'calls':expected_calls, 'input_tokens':sum(r['input_tokens'] for r in ordered),
            'output_tokens':sum(r['output_tokens'] for r in ordered),
            'query_encoder_tokens':0 if arm=='Raw_MV8' else metadata['query_encoder_tokens'][question],
            'invalid_calls':sum(r['answer_class'] is None for r in ordered),
            'truncated_calls':sum(r['truncated'] for r in ordered)}
        reconstructed['processed_tokens'] = sum(reconstructed[k] for k in
            ('input_tokens', 'output_tokens', 'query_encoder_tokens'))
        if reconstructed != expected[question, arm]:
            raise ValueError('Vote or cost reconstruction mismatch')
    report = {'call_observations':len(calls), 'distinct_cached_requests':len(physical),
        'reused_observations':len(calls)-len(physical), 'votes_checked':len(groups),
        'distinct_request_input_tokens':sum(r['input_tokens'] for r in physical.values()),
        'distinct_request_output_tokens':sum(r['output_tokens'] for r in physical.values()),
        'new_generation_calls':0, 'independent_regrading':False}
    if report != metadata['expected']:
        raise ValueError('Cached request accounting mismatch')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.outcomes.read_text().splitlines() if line.strip()]
    values = aggregate(rows)
    verify_reference(values, json.loads(args.reference.read_text()))
    report = {"state": "passed_recorded_outcome_reconstruction",
              "records": len(rows), "cohorts": values,
              "fresh_generation": False, "independent_regrading": False,
              "accepted_paper_full_benchmark_reproduction": False}
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as f:
            f.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
