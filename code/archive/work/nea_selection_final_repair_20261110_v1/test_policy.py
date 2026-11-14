import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'nea_spo_feedback_allocation_20261108_v1'))
sys.path.insert(0,str(ROOT))
from policy import MarginalAllocator, RetryOrder

class Tests(unittest.TestCase):
    def test_marginal_telescopes(self):
        a=MarginalAllocator()
        for cost in [.9,.9,.8,1.]:
            a.choose(('parent',),[0.],0,1.)
            a.observe([cost],[True])
        self.assertAlmostEqual(a.states[('parent',)]['sum'][0],.2)
        self.assertEqual(a.states[('parent',)]['completed'],[4])
        self.assertEqual(a.log[1]['reward'],0.)

    def test_invalid_no_reward(self):
        a=MarginalAllocator();a.choose(('p',),[1.],0,1.)
        a.observe([.1],[False]);self.assertEqual(a.log[0]['reward'],0.)

    def test_retry_only_same_parent(self):
        a=RetryOrder(); key=((1,2),)
        event={'parent_routes':[[1,2]],'j':0,'parent_cost':1.,'candidate_cost':1.,'valid':True}
        self.assertEqual(a.choose(key,[2,1]),0)
        a.observe([event]);self.assertEqual(a.choose(key,[2,1]),0)
        a.observe([event]);self.assertEqual(a.choose(key,[2,1]),1)
        self.assertEqual(a.choose(((2,3),),[2,1]),0)
        event['candidate_cost']=.9;a.observe([event])
        self.assertEqual(a.choose(key,[2,1]),0)

if __name__=='__main__':unittest.main()
