"""Score-driven call allocation. No generated answers or labels enter planning."""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import itertools
import math
import random
from typing import Iterable
from maze.io import identity


@dataclass(frozen=True)
class Candidate:
    path: tuple[str, ...]
    score: float

    def __post_init__(self):
        if not self.path or any(not isinstance(v, str) or not v for v in self.path) or len(set(self.path)) != len(self.path):
            raise ValueError("A candidate is a nonempty ordered tuple of distinct exemplar IDs")
        if isinstance(self.score, bool) or not isinstance(self.score, (float, int)) or not math.isfinite(self.score):
            raise ValueError("Candidate score must be finite")

    @property
    def path_id(self) -> str:
        return identity(list(self.path))


@dataclass(frozen=True)
class RequestSlot:
    slot: int
    path: tuple[str, ...]
    seed: int
    pool: int | None = None


def sample_paths(bank_ids: Iterable[str], shots: int, pool_size: int, seed: int) -> list[tuple[str, ...]]:
    ids = sorted(bank_ids)
    if len(ids) != len(set(ids)) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("Bank IDs must be distinct nonempty strings")
    if type(shots) is not int or not 1 <= shots <= len(ids):
        raise ValueError("shots must fit the eligible bank")
    if type(pool_size) is not int or pool_size < 1 or type(seed) is not int:
        raise ValueError("Positive pool size and integer seed required")
    count = math.perm(len(ids), shots)
    if pool_size > count:
        raise ValueError("Requested more distinct paths than the bank permits")
    rng = random.Random(seed)
    if count <= 100_000 and pool_size > count // 4:
        return rng.sample(list(itertools.permutations(ids, shots)), pool_size)
    result, seen = [], set()
    while len(result) < pool_size:
        path = tuple(rng.sample(ids, shots))
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


def _rank(pool: Iterable[Candidate], anchor: tuple[str, ...] | None = None) -> list[Candidate]:
    candidates = list(pool)
    if not candidates or len({c.path for c in candidates}) != len(candidates):
        raise ValueError("A pool needs distinct complete paths")
    if anchor is not None and anchor not in {c.path for c in candidates}:
        raise ValueError("The specified anchor must be present in the pool")
    return sorted(candidates, key=lambda c: (-float(c.score), c.path != anchor, c.path_id))


def allocate(pools: list[list[Candidate]], policy: str, calls: int, *, seed: int = 0,
             anchor: tuple[str, ...] | None = None,
             repetitions: list[int] | None = None) -> list[RequestSlot]:
    """Allocate physical requests; repeated winners are never deduplicated.

    `independent` uses one winner per supplied pool. `topk` uses distinct paths
    from a shared pool. `repeat` assigns every call to its single best path.
    `allocation` assigns explicit repetition counts to the ranked shared paths.
    """
    if type(calls) is not int or calls < 1 or type(seed) is not int or not pools:
        raise ValueError("A positive call budget, integer seed and nonempty pools are required")
    if policy == "independent":
        if len(pools) != calls or repetitions is not None:
            raise ValueError("Independent policy requires exactly one pool per request")
        selections = [(i, _rank(pool, anchor)[0].path) for i,pool in enumerate(pools)]
    else:
        if len(pools) != 1:
            raise ValueError("This policy uses a single shared pool")
        ranked = _rank(pools[0], anchor)
        if policy == "repeat":
            selections = [(0, ranked[0].path)] * calls
        elif policy == "topk":
            if calls > len(ranked):
                raise ValueError("Insufficient distinct paths for top-k")
            selections = [(0, c.path) for c in ranked[:calls]]
        elif policy == "allocation":
            if (not repetitions or len(repetitions) > len(ranked)
                    or any(type(n) is not int or n < 1 for n in repetitions)
                    or sum(repetitions) != calls):
                raise ValueError("Positive repetition counts must sum to the call budget")
            selections = [(0, ranked[j].path) for j,n in enumerate(repetitions) for _ in range(n)]
        else:
            raise ValueError(f"Unknown allocation policy: {policy}")
        if repetitions is not None and policy != "allocation":
            raise ValueError("repetitions is only valid for explicit allocation")
    return [RequestSlot(i, path, seed + i, pool) for i,(pool,path) in enumerate(selections)]


def vote(answers: Iterable[str | None], *, seed: int = 0, tie: str = "seeded") -> str | None:
    values = list(answers)
    if any(a is not None and not isinstance(a, str) for a in values):
        raise ValueError("Voting accepts parsed strings or None")
    valid = [a for a in values if a is not None and a.strip()]
    if not valid:
        return None
    counts = Counter(valid)
    winners = {a for a,n in counts.items() if n == max(counts.values())}
    if tie == "first":
        return next(a for a in valid if a in winners)
    if tie == "seeded":
        return random.Random(seed).choice(sorted(winners))
    raise ValueError("Tie rule must be first or seeded")
