# The Maze method

Maze separates three decisions: which exemplar path to consider, how to score the complete path, and how to spend the generation budget after selection.
The target model is fixed.
A separate encoder supplies the representations used by the planner.

## Ordered prompt trajectories

For a query $q$ and training exemplar bank, an ordered path $P=[e_{\pi(1)},\ldots,e_{\pi(K)}]$ becomes a complete prompt from the system instruction through the exemplars to the query.
Node features may encode full exemplars or question text alone; the renderer still supplies complete exemplars to the target.
The system node is explicitly encoded in the full-QA configuration and represented by zero in the compact question-node configuration.

## Directed path energy

For a view $\ell$, a low-rank relation projection produces directed transition features.
In the shared-projection implementation,

$$
\mathbf d_{i,i+1}^{(\ell)} = \mathbf z_i^{(\ell)}-\mathbf z_{i+1}^{(\ell)}
+\tanh\left(\sum_{j\in\mathcal N(\ell)}\left[\mathbf z_i^{(j)}-\mathbf z_{i+1}^{(\ell)}\right]\right).
$$

The source-node mass modulates each transition before accumulation:

$$
\mathbf E_P^{(\ell)}=\sum_{i=0}^{K}\tanh\left(\mathbf m_i^{(\ell)}\odot\mathbf d_{i,i+1}^{(\ell)}\right).
$$

Changing exemplar order changes the directed edges and their source-node weights.
This is not an unweighted displacement sum that collapses to the endpoints.

Query-conditioned fusion weights each view's path energy and its neighbors.
Scalar fusion sums coordinates before the cross-view nonlinearity; vector fusion retains them until the final readout.
These are distinct configurations of the same structured construction, not algebraically interchangeable formulas.

| Setting | Scalar configuration | Compact vector | Full-QA vector |
|---|---|---|---|
| Node/view normalization | None in the equation reference | Row-local norm, scaled by $\sqrt d$ | Same row-local normalization |
| Path representation | $r$-vector | $r$-vector | $r$-vector |
| Cross-view fusion | Scalar channel sum | Coordinate-preserving | Coordinate-preserving |
| Fusion boundary | Interior cross-view terms; boundary own-view terms | Clip neighbors to available views | All views supply context; only interior views read out |
| Final readout | Identity scalar sum | Learned linear readout | Learned linear readout |
| Example view count | 32 | 4 | 32 |

The window size, representation dimension, path length, projection rank, and fusion width are configurable.
The compact 32-view recipe uses views 3–30 for readout in one-based notation.
The tests compare NumPy and PyTorch implementations with an independent scalar reference and the preserved four-view and 32-view compact implementations.

## Offline learning

Correctness comes from the target model's evaluated responses, not from the frozen encoder.
The planner is fitted on cached representations using success prediction and within-query comparisons.
Local content replacements and order swaps provide focused comparisons; a fixed reference prompt provides a stable comparison point for each query.
For binary preference pairs, the logistic objective is

$$
\ell_{\mathrm{pair}}=\log(1+\exp(s^{-}-s^{+})).
$$

The generic trainer also supports reference-relative soft targets $(1+y_i-y_a)/2$ for empirical correctness means.
It averages within queries and then gives each training query equal weight.
Validation query identities are disjoint from fitting queries.
The generic trainer exports only the success/ranking parameters it fits; it does not mark auxiliary harm or cost heads as trained.
Original auxiliary-loss definitions and compatible numerical replay remain available under [reproduction/compact](../reproduction/compact/).

## MultiMaze allocation

After ranking, selected trajectories receive $n_j$ physical requests with $\sum_j n_j=N$.
A repeated winner is not deduplicated: it still consumes a separate request and contributes a separate response.

| Policy | Planner behavior |
|---|---|
| `repeat` | One search, one selected path, $N$ requests |
| `topk` | One pool, top-$N$ distinct paths, one request each |
| `independent` | $N$ pools, one selected winner per pool |
| `allocation` | One pool, explicit positive repetition counts on ranked paths |

All requests and decoding seeds are fixed before generation.
The query representation is reused across selections and repetitions.
Majority voting is a downstream operation over responses; it does not alter prompt selection.

## Extending the scorer

Consensus combines independently fitted energy scores without additional target calls.
Residual calibration fits reference-relative utilities from input-side forecasts on disjoint fitting queries.
New scorers can use the same candidate and allocation interfaces, provided their predictions do not consume evaluation answers or observed response costs.
These interfaces permit ongoing Maze research without conflating a new configuration with an earlier experiment.
