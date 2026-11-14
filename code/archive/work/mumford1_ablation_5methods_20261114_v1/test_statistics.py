import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as a


class StatisticsTests(unittest.TestCase):
    def test_holm_family_order_and_monotonicity(self):
        self.assertEqual(a.holm([.04, .001, .03]), [.06, .003, .06])
        self.assertEqual(a.holm([.9, .8, .7]), [1., 1., 1.])

    def test_win_tie_loss_uses_absolute_tolerance(self):
        self.assertEqual(a.outcome([2e-6, 1e-7, -1e-7, -2e-6]),
                         dict(wins=1, ties=2, losses=1))

    def test_exact_rank_probability(self):
        audit = a.e.module('rank_only_reference', a.e.PREVIOUS / 'analyze.py')
        self.assertEqual(audit.signed_rank_exact(list(range(1, 11)))['p_two_sided'], 2/1024)
        self.assertEqual(audit.signed_rank_exact([0.] * 10)['p_two_sided'], 1.)
        self.assertEqual(audit.signed_rank_exact([1., -1.])['p_two_sided'], 1.)


if __name__ == '__main__':
    unittest.main()
