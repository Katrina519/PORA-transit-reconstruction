import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from labels import aggregate


class LabelTests(unittest.TestCase):
    def rows(self):
        return [{'j':0,'seed':s,'feasible':True,'parent_cost':1.,'cost':c} for s,c in [(1,.8),(2,1.1),(3,.9)]]+[
            {'j':1,'seed':s,'feasible':False,'parent_cost':1.,'cost':.1} for s in [1,2,3]]

    def test_uses_mean_of_all_trials_and_zero_for_invalid(self):
        a=aggregate(self.rows(),2,[1,2,3])
        self.assertAlmostEqual(a[0],.1)
        self.assertEqual(a[1],0.)

    def test_missing_or_duplicate_trials_are_rejected(self):
        with self.assertRaises(ValueError):aggregate(self.rows()[:-1],2,[1,2,3])
        with self.assertRaises(ValueError):aggregate(self.rows()+[self.rows()[0]],2,[1,2,3])


if __name__=='__main__':unittest.main()
