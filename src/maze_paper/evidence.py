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
