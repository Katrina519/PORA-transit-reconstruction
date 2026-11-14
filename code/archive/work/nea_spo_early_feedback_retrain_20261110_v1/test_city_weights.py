import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from city_weights import weights


class CityWeightsTests(unittest.TestCase):
    def test_city_total_stays_equal_with_different_gain_scales(self):
        result=weights([10.,20.,.1,.2],['a','a','b','b'])
        self.assertAlmostEqual(sum(result[:2]),.5)
        self.assertAlmostEqual(sum(result[2:]),.5)

    def test_zero_gain_state_gets_no_artificial_reward_weight(self):
        result=weights([0.,2.,3.],['a','a','b'])
        self.assertEqual(result[0],0.)
        self.assertAlmostEqual(result[1],.5)
        self.assertAlmostEqual(result[2],.5)


if __name__=='__main__':unittest.main()
