"""Input-side allocation and explicit voting contracts; no model/API client."""
import math
import random
from collections import Counter


def _rank(pool):
    items = list(pool)
    if not items:
        raise ValueError("candidate pool is empty")
    ids = []
    for item in items:
        if set(item) != {"path_id", "score"}:
            raise ValueError("candidate fields must be exactly path_id and score")
        if not isinstance(item["path_id"], str) or not item["path_id"]:
            raise ValueError("path_id must be a nonempty string")
        score = item["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("score must be a finite number")
        ids.append(item["path_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate path identity within a pool")
    # The manuscript does not prescribe score-tie resolution; this reference
    # chooses lexical path identity and requires it to be frozen in a run.
    return sorted(items, key=lambda x: (-x["score"], x["path_id"]))


def shared_pool_topk(pool, k):
    ranked = _rank(pool)
    if type(k) is not int or not 1 <= k <= len(ranked):
        raise ValueError("k must be an integer within the pool")
    return [{"request_slot": i, "path_id": row["path_id"]}
            for i, row in enumerate(ranked[:k])]


def independent_pool_winners(pools):
    pools = list(pools)
    if not pools:
        raise ValueError("at least one independent pool is required")
    # Repeated winners remain separate requests; never deduplicate physical calls.
    return [{"request_slot": i, "pool_index": i, "path_id": _rank(pool)[0]["path_id"]}
            for i, pool in enumerate(pools)]


def majority_vote(answers, seed):
    if type(seed) is not int:
        raise ValueError("a frozen integer seed is required")
    values = list(answers)
    if any(v is not None and not isinstance(v, str) for v in values):
        raise ValueError("answers must be parsed strings or None")
    counts = Counter(v for v in values if v is not None and v.strip())
    if not counts:
        return None
    tied = sorted(v for v, n in counts.items() if n == max(counts.values()))
    return random.Random(seed).choice(tied)


def validate_disjoint_question_keys(train, validation, test):
    """Caller supplies normalized-question hashes, not pair/response IDs."""
    groups = [list(x) for x in (train, validation, test)]
    if any(any(not isinstance(v, str) or not v for v in g) for g in groups):
        raise ValueError("normalized question keys must be nonempty strings")
    if any(len(g) != len(set(g)) for g in groups):
        raise ValueError("duplicate normalized question within a split")
    a, b, c = map(set, groups)
    if a & b or a & c or b & c:
        raise ValueError("question leakage across splits")
    return {"train": len(a), "validation": len(b), "test": len(c)}
