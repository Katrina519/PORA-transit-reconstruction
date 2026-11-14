import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment as e


class NoGainTests(unittest.TestCase):
    def test_observed_gain_is_recorded_but_not_used_for_selection(self):
        allocator = e.NoGainAllocator(.05)
        key = ((0, 1), (1, 2))
        self.assertEqual(allocator.choose(key, [0., 0.], 7, 1.), 0)
        costs = [1.] * 10
        valid = [False] * 10
        costs[7], valid[7] = .1, True
        allocator.observe(costs, valid)
        self.assertAlmostEqual(allocator.states[key]['sum'][0], .9)
        self.assertAlmostEqual(allocator.log[0]['reward'], .9)
        self.assertEqual(allocator.states[key]['completed'], [1, 0])
        self.assertEqual(allocator.choose(key, [0., 0.], 2, 1.), 1)

    def test_parent_history_is_isolated_and_pending_counts_are_updated(self):
        allocator = e.NoGainAllocator(.05)
        a, b = ((0, 1), (1, 2)), ((0, 2), (2, 1))
        self.assertEqual(allocator.choose(a, [0., 0.], 7, 1.), 0)
        self.assertEqual(allocator.choose(a, [0., 0.], 2, 1.), 1)
        self.assertEqual(allocator.choose(b, [0., 0.], 4, 1.), 0)
        self.assertEqual(allocator.states[a]['visits'], [1, 1])
        self.assertEqual(allocator.states[a]['completed'], [0, 0])
        allocator.observe([1.] * 10, [False] * 10)
        self.assertEqual(allocator.states[a]['completed'], [1, 1])
        self.assertEqual(allocator.states[b]['completed'], [1, 0])
        self.assertFalse(allocator.pending)

    def test_full_keeps_the_gain_term(self):
        allocator = e.Allocator(.05)
        key = ((0, 1), (1, 2))
        allocator.choose(key, [0., 0.], 0, 1.)
        allocator.observe([.1], [True])
        self.assertEqual(allocator.choose(key, [0., 0.], 0, 1.), 0)


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pilot, cls.runtime = e.load_runtime()

    def test_frozen_full_and_ridge_score_callbacks(self):
        import numpy as np
        models = e.documents()
        x = np.arange(150, dtype=float).reshape(15, 10) / 100.
        for name in ['full', 'ridge']:
            doc = models[name]['mumford0']
            e.configure(self.pilot, self.runtime, doc)
            callback = self.runtime.d.b.base.BASE_SELECTION.__globals__['joint_score']
            np.testing.assert_allclose(callback(x),
                self.runtime.d.b.learner.scores(doc, x, 'mumford0'), rtol=0, atol=1e-12)
            self.assertEqual(self.runtime.d.q.ROADS.shape, (70, 70))
            self.assertEqual(self.runtime.d.b.base.SPEC['routes'], 15)

    def test_method_flags_and_models(self):
        for name in e.METHODS:
            fn = e.make(self.pilot, self.runtime, name)
            self.assertEqual(fn.__globals__['online'], name in ['full', 'ridge', 'nogain'])
            self.assertEqual(fn.__globals__['frozen']('mumford1'),
                e.documents()['ridge' if name == 'ridge' else 'full']['mumford0'])
            if name == 'nogain':
                alloc = fn.__globals__['BatchAllocator'](.05, repair=False)
                self.assertFalse(alloc.use_feedback)
                self.assertTrue(alloc.use_prior)

    def test_count_balances_fifteen_targets(self):
        policy = self.runtime.CountPolicy()
        key = ((0, 1), (1, 2))
        self.assertEqual([policy.choose(key, 15) for _ in range(30)], list(range(15)) * 2)

    def test_request_order_and_frozen_sources(self):
        req = e.read(e.ROOT / '请求冻结.json')
        from collections import Counter
        self.assertEqual(len(req['tasks']), 50)
        self.assertEqual(len({(t['seed'], t['variant']) for t in req['tasks']}), 50)
        self.assertEqual(req['seeds'], list(range(3011, 3021)))
        self.assertEqual(set(Counter((t['variant'], t['position']) for t in req['tasks']).values()), {2})
        e.check(req['hashes'])


if __name__ == '__main__':
    unittest.main()
