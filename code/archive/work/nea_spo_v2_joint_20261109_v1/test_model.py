"""验证实例尺度、城市等权及SPO+任务后悔上界。"""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import model
import numpy as np


class ModelTests(unittest.TestCase):
    def test_city_weights_are_half_despite_different_state_counts(self):
        p = model.probabilities(['mandl', 'mandl', 'mumford0'])
        self.assertAlmostEqual(float(p[:2].sum()), .5)
        self.assertAlmostEqual(float(p[2]), .5)

    def test_scaled_encoding_has_instance_interpretable_values(self):
        raw = [.1, 1.875, .2, .3, .1, .2, .3, .8, .5, .1]
        spec = {'routes': 12, 'diameter': 26., 'max': 15, 'n': 30}
        out = model.encode(raw, spec, 'scaled')
        self.assertAlmostEqual(out[0], 42./312)
        self.assertAlmostEqual(out[1], 1.)
        self.assertAlmostEqual(out[2], .1)
        self.assertAlmostEqual(out[4], 3.3/26.)

    def test_scaled_encoding_is_invariant_to_time_units(self):
        raw = np.array([.1, 1.875, .2, .3, .1, .2, .3, .8, .5, .1])
        changed = raw.copy(); changed[[0,4,7,8,9]] *= 60
        spec = {'routes': 12, 'diameter': 26., 'max': 15, 'n': 30}
        other = dict(spec, diameter=1560.)
        np.testing.assert_allclose(model.encode(raw, spec, 'scaled'), model.encode(changed, other, 'scaled'))

    def test_spo_loss_upper_bounds_regret_of_highest_score(self):
        for gains in [[0.,1.,.4], [0.,0.,0.], [1.,1.,.5]]:
            for scores in [[0.,1.,.4], [1.,0.,.3], [-.5,.6,2.]]:
                chosen = int(np.argmax(scores))
                self.assertGreaterEqual(model.loss(scores,gains)+1e-12, max(gains)-gains[chosen])


if __name__ == '__main__':
    unittest.main()
