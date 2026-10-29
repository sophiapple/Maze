# Maze compact-memory and adaptive-inference variants

**These variants belong to the Maze project and retain the Maze name.**
They extend the NeurIPS idea of choosing reusable input-side reasoning support and spending target calls selectively. Repeated decoding of a selected prompt is already part of MultiMaze. Compact trajectory excerpts, source-competence selection and answer-agreement stopping are additional choices in this research configuration.

The selected observations below demonstrate useful accuracy–token trade-offs against named voting controls. They are exposed exploratory results on Qwen3.5-9B, not results reported in the NeurIPS paper, an equation-equivalent reproduction of its scorer, or an independent confirmation of a universal method.

## Relationship to NeurIPS Maze

| Component | NeurIPS Maze | Compact-memory research configuration |
|---|---|---|
| Reusable support | Complete ordered exemplars | Bounded excerpts of successful training-source trajectories, with provenance and a warning that the source is not guaranteed correct for the query |
| Input-side selection | Learned geometric energy over an ordered path | Frozen source competence, cached semantic relevance, category/role eligibility and explicit fallback rules; these highlights do not test the learned geometric scorer |
| Multiple calls | MultiMaze allocations, including repeated decoding of one selected prompt | Raw first response, then selected memory; repeat the primary or use a complement according to the named policy |
| Stopping | Allocation fixed before generation in the paper setup | Stop after two equal, nonnull parsed answers; otherwise take the third and apply literal voting with earliest-valid tie resolution |
| Model boundary | Text-in/text-out target, separate representations | Same black-box target boundary; cached query/source representations are reused |
| Attribution | Cite the NeurIPS paper for Maze's framework | Also identify the compact-memory policy, model, cohort and exploratory protocol when citing these measurements |

This is a **Maze-family method variant**, with changes beyond a numerical or backend implementation of the paper equations. The repository's existing scorer implementations and paper reproduction remain their own stable reference. This documentation update leaves the software package version at 0.2.1.

## Policies behind the highlights

- **Repeated primary:** acquire a Raw answer, a primary-memory answer, and, if required by simulated agreement stopping, another sample of the same primary-memory prompt. Reusing the prompt is not reusing the generated answer.
- **Support switch:** use the complementary-memory policy when two distinct usable memories exist; otherwise use Raw adaptive voting. The input-support rule was frozen before this outcome join.
- **Role-repeat:** retain an available same-category primary. When fewer than two same-category sources have fixed competence at least 0.5 and positive cosine, allow same-benchmark candidates with at least 0.5 query-role coverage, with the frozen ranking, exclusions and unknown-role fallback. Repeat the selected primary. Role cues are coarse input features, not neural groups or certificates of semantic correctness.
- **Benchmark portfolio:** apply Raw adaptive on GPQA and role-repeat on MMLU-Pro/BBEH. This is one outcome-informed benchmark-ID rule retaining all 171 questions, not a learned universal router or per-question correctness routing.

For a Raw/memory prefix, exact nonnull agreement returns the common answer. Otherwise the third response participates in literal parsed-answer voting; ties choose the earliest valid answer. A third answer cannot overturn two identical votes. This is a property of the voting rule, not a new empirical discovery.

Memory rendering uses the frozen compact-excerpt algorithm and source offsets, a 1,600-token memory budget including the warning/source question, and Raw fallback on unusable support. The protocol bounds full input to 8,192 tokens and prospective output to 16,384 within a 32,768-token context. These are protocol descriptions; this documentation release does not ship the private study runner, source trajectories, model outputs or question-level campaign archives.

## Selected positive comparisons

<!-- BEGIN COMPACT MEMORY HIGHLIGHTS -->
| Panel | Maze variant | Correct / questions | Cached control | Control correct / questions | Fewer simulated target tokens |
|---|---|---:|---|---:|---:|
| MMLU-Pro64 | Repeated primary | **58/64** | Raw3 | 56/64 | **23.56%** |
| MMLU-Pro64 | Repeated primary | **58/64** | Raw8 | 56/64 | **70.61%** |
| BBEH43 | Support switch | **16/43** | Raw3 | 15/43 | **11.17%** |
| BBEH43 | Role-repeat (conditional mean) | **16.5/43 (mean)** | Raw3 | 15/43 | **4.62%** |
| Full exposed panel171 | Benchmark portfolio (conditional mean) | **119.5/171 (mean)** | Raw3 | 116/171 | **15.07%** |
| Full exposed panel171 | Benchmark portfolio (conditional mean) | **119.5/171 (mean)** | Raw8 | 119/171 | **68.87%** |
<!-- END COMPACT MEMORY HIGHLIGHTS -->

Raw3 and Raw8 are cached three- and eight-answer voting controls. The two rows for the same policy against Raw3/Raw8 share observations. They are not separate replications. MMLU-Pro64 covers four science/math categories; BBEH43 spans 21 categories. The 171-question panel contains GPQA64, MMLU-Pro64 and BBEH43.

**Replicates and exposure.** Role-repeat's BBEH scores are 17/43 and 16/43. Only 15 input-defined BBEH scarcity cases have fresh second/third responses; the other 156 questions in that study reuse cached observations. Its incremental fresh recovery is one question in one conditional replicate. The portfolio scores 120/171 and 119/171 and reuses those same outcomes: its 156 unchanged questions are duplicated across replicates, not independently reevaluated. Repeated-primary was emphasized after observing the screen, and the portfolio rule was chosen using earlier outcomes. Matched-seed follow-ups show seed sensitivity; they do not establish a consistent advantage over matched Raw continuation. A separate full-panel fresh-draw study is pending and contributes no accuracy result here.

**Comparison scope.** These are selected positive highlights, not a full benchmark leaderboard. They do not establish dominance over every control or across benchmarks. Raw adaptive remains cheaper on the highlighted panels, and the BBEH memory points remain below cached Raw8. Role/no-gate aliases cannot identify a role-gate benefit. Exact original-Maze and strongest external compact-memory asset/protocol comparisons remain open. No broad-superiority or internal-neural-mechanism claim is made.

**Uncertainty.** Previously computed, unadjusted 95% paired question-cluster gain intervals are [0, 7.81] percentage points for parent MMLU-Pro versus Raw voting; [0, 6.98] for BBEH support-switch versus Raw3; and [0, 9.30] for BBEH role-repeat versus Raw3. For the portfolio, they are [0.29, 4.39] versus Raw3 and [-2.92, 3.51] versus Raw8. Conditional replicates stay together within question clusters. These intervals do not adjust for outcome-informed search and do not establish confirmatory significance. Aggregate highlights alone cannot reconstruct the intervals.

## Cost and reproducibility scope

The displayed reduction is

```text
100 * (1 - mean simulated policy target tokens / mean comparator target tokens)
```

Tokens include the input and output of consumed simulated prefixes and every full prompt reread. All physical research acquisitions, including thirds unused by simulated stopping, are charged separately. These percentages exclude source construction, encoder/tokenizer/planner/parser/audit overhead and are not measured online latency, monetary savings or physically avoided calls. They use a different accounting denominator from the older four-exemplar development tables, which also include once-per-query encoder processing.

The completed parent, prefix and role acquisitions cost 3,331 physical target calls / 16,529,625 tokens. Source-parent observations add 2,106 / 14,162,433 tokens; query encoding adds 296 queries / 4,406 forwards / 228,752 encoder tokens / 59.7738 wall seconds. Other preparation, rendering and audit overhead is additional. Amortization requires an explicit future-query count; GPU tariff and some historical overhead remain unresolved. API spend zero does not make local compute free.

The [public aggregate highlights](../evidence/highlights/compact_memory.json) contain only the selected comparison totals, cost denominators, reported intervals, scope and hashes binding the private source summaries. They include no benchmark questions, answers, generated traces, source texts, credentials or unsuccessful experiment datasets. Existing published development records remain unchanged.

```bash
python scripts/check_memory_highlights.py
python scripts/verify_release.py
```

The first command verifies arithmetic and table consistency on CPU, without dependencies or network access. It does **not** rerun the model, replay votes, regrade answers, recompute paired intervals or independently validate private summaries. The private research study used separate same-controller mechanical reconstructions; those are not external or semantic audits. The second command additionally verifies the repository's previously published development and paper evidence under their original reproduction contracts.

Use the [NeurIPS citation](../README.md#citation) for the original Maze framework and identify this named variant and protocol separately. A compact-memory result must not be relabelled as a paper experiment or attributed to every Maze scorer.
