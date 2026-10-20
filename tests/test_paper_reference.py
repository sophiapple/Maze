import math
import unittest

import numpy as np

from maze_paper.scalar_energy import ScalarWeights, score_paths, preference_loss
from maze_paper.policies import (shared_pool_topk, independent_pool_winners,
                                majority_vote, validate_disjoint_question_keys)


def literal_score(path, weights):
    """Separate scalar-loop reconstruction of the printed equations."""
    wmu, wm, wh, wo = [a.tolist() for a in weights.checked()]
    p = path.tolist()
    nodes, layers, dim = path.shape
    rank, hidden = len(wmu[0]), len(wh[0])
    z = [[[sum(p[i][l][a] * wmu[a][b] for a in range(dim))
           for b in range(rank)] for l in range(layers)] for i in range(nodes)]
    m = [[[sum(p[i][l][a] * wm[a][b] for a in range(dim))
           for b in range(rank)] for l in range(layers)] for i in range(nodes)]
    ep = []
    for l in range(layers):
        vector = []
        for b in range(rank):
            value = 0.0
            for i in range(nodes - 1):
                cross = sum(z[i][j][b] - z[i+1][l][b]
                            for j in range(layers) if 0 < abs(j-l) <= 2)
                delta = z[i][l][b] - z[i+1][l][b] + math.tanh(cross)
                value += math.tanh(m[i][l][b] * delta)
            vector.append(value)
        ep.append(vector)
    def u(l, j):
        inp = m[-1][l] + ep[j]
        hid = [math.tanh(sum(inp[a]*wh[a][b] for a in range(2*rank)))
               for b in range(hidden)]
        return sum(sum(hid[a]*wo[a][b] for a in range(hidden))*ep[j][b]
                   for b in range(rank))
    total = 0.0
    for l in range(layers):
        total += u(l, l)
        if l in range(2, layers-2):
            total += math.tanh(sum(u(l, j) for j in range(layers)
                                   if 0 < abs(j-l) <= 2))
    return total


class ScalarEquationTests(unittest.TestCase):
    def setUp(self):
        g = np.random.default_rng(20261020)
        self.x = g.normal(size=(3, 6, 7, 5))
        self.w = ScalarWeights(g.normal(scale=.25, size=(5, 3)),
                               g.normal(scale=.25, size=(5, 3)),
                               g.normal(scale=.25, size=(6, 4)),
                               g.normal(scale=.25, size=(4, 3)))

    def test_independent_literal_reconstruction_across_boundaries(self):
        for layers in (1, 2, 4, 5, 7):
            with self.subTest(layers=layers):
                x = self.x[:, :, :layers]
                np.testing.assert_allclose(score_paths(x, self.w),
                    [literal_score(p, self.w) for p in x], atol=1e-12, rtol=1e-12)

    def test_batch_and_single_path_parity(self):
        np.testing.assert_allclose(score_paths(self.x, self.w),
            [score_paths(x[None], self.w)[0] for x in self.x])

    def test_exemplar_order_is_active(self):
        changed = self.x[:, [0, 4, 3, 2, 1, 5]]
        self.assertFalse(np.allclose(score_paths(self.x, self.w), score_paths(changed, self.w)))

    def test_query_conditioning_is_active(self):
        x = self.x.copy()
        x[:, -1] = 0
        self.assertFalse(np.allclose(score_paths(x, self.w), score_paths(self.x, self.w)))

    def test_zero_source_states_have_zero_outgoing_energy(self):
        x = self.x.copy()
        x[:, :-1] = 0
        np.testing.assert_array_equal(score_paths(x, self.w), np.zeros(3))

    def test_no_implicit_node_normalization(self):
        self.assertFalse(np.allclose(score_paths(self.x, self.w),
                                     score_paths(self.x * 2, self.w)))

    def test_bias_free_parameter_count(self):
        w = ScalarWeights(np.zeros((2560,64)), np.zeros((2560,64)),
                          np.zeros((128,96)), np.zeros((96,64)))
        self.assertEqual(w.parameter_count, 2*2560*64+3*64*96)
        self.assertEqual(w.parameter_count, 346112)

    def test_reject_bad_shapes(self):
        for x in (self.x[0], self.x[:, :1], self.x[:, :, :, :4], self.x[:0]):
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                score_paths(x, self.w)

    def test_reject_nonfinite(self):
        for value in (float("nan"), float("inf")):
            x = self.x.copy(); x[0, 0, 0, 0] = value
            with self.assertRaises(ValueError):
                score_paths(x, self.w)

    def test_bad_weight_contract(self):
        bad = ScalarWeights(self.w.relation, self.w.mass, np.zeros((5,4)), self.w.fusion_output)
        with self.assertRaises(ValueError):
            score_paths(self.x, bad)

    def test_logistic_loss_sign_and_stability(self):
        self.assertLess(preference_loss([3.0], [-3.0]), preference_loss([-3.0], [3.0]))
        self.assertAlmostEqual(preference_loss([0.0], [0.0]), math.log(2))
        self.assertAlmostEqual(preference_loss([-1000.0], [1000.0]), 2000.0)

    def test_loss_rejects_unpaired_and_nonfinite(self):
        for a,b in (([],[]),([0],[0,1]),([float("nan")],[0])):
            with self.assertRaises(ValueError):
                preference_loss(a,b)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.pool = [{"path_id": "b", "score": 2.0},
                     {"path_id": "a", "score": 2.0}, {"path_id": "c", "score": 1.0}]

    def test_shared_distinct_topk(self):
        self.assertEqual([x["path_id"] for x in shared_pool_topk(self.pool, 2)], ["a","b"])

    def test_independent_repeated_winner_preserves_calls(self):
        plan = independent_pool_winners([self.pool, self.pool])
        self.assertEqual(len(plan), 2)
        self.assertEqual([r["path_id"] for r in plan], ["a","a"])
        self.assertNotEqual(plan[0]["request_slot"], plan[1]["request_slot"])

    def test_selection_rejects_outcome_features(self):
        row = dict(self.pool[0], correct=True)
        with self.assertRaises(ValueError):
            shared_pool_topk([row], 1)

    def test_pool_contract(self):
        for pool,k in (([],1),(self.pool,0),(self.pool,4),(self.pool,True),
                       ([self.pool[0],self.pool[0]],1),
                       ([{"path_id":"x","score":float("inf")}],1)):
            with self.assertRaises(ValueError):
                shared_pool_topk(pool,k)
        with self.assertRaises(ValueError):
            independent_pool_winners([])

    def test_seeded_ties_and_invalid_votes(self):
        self.assertEqual(majority_vote(["A","B",None,""],17), majority_vote(["A","B"],17))
        self.assertEqual(majority_vote(["B","A","B"],1), "B")
        self.assertIsNone(majority_vote(["", None],1))
        self.assertEqual({majority_vote(["A","B"],i) for i in range(32)}, {"A","B"})

    def test_invalid_vote_contract(self):
        for answers,seed in (([True],1),(["A"],None)):
            with self.assertRaises(ValueError):
                majority_vote(answers,seed)

    def test_disjoint_question_identity(self):
        self.assertEqual(validate_disjoint_question_keys(["a"],["b"],["c"])["test"], 1)
        for a,b,c in ((["a"],["b"],["a"]),(["a","a"],[],[]),([None],[],[])):
            with self.assertRaises(ValueError):
                validate_disjoint_question_keys(a,b,c)


if __name__ == "__main__":
    unittest.main()
