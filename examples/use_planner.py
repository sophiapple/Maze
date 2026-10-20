"""Inspect generalized call allocation using invented scores."""
from maze import Candidate, allocate

paths = [
    Candidate(("example-a", "example-b", "example-c", "example-d"), 2.4),
    Candidate(("example-b", "example-a", "example-c", "example-d"), 1.7),
    Candidate(("example-c", "example-b", "example-a", "example-d"), 1.2),
]

# Four physical calls: two for the highest-scoring path, then one for each other path.
requests = allocate([paths], "allocation", calls=4, repetitions=[2, 1, 1], seed=42)
for request in requests:
    print(request.slot, request.path, request.seed)
