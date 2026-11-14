import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from learner import fit_multi


class FitTests(unittest.TestCase):
    def fixture(self):
        prior={'encoding':'raw','model':{'indices':list(range(10)),'mean':[0.]*10,'scale':[1.]*10,
               'coef':[0.,0.,.7]+[0.]*8,'lambda':.1}}
        tasks=[{'state':'s','j':j,'x':[float(j)]+[0.]*9,'gain':float(j)} for j in range(3)]
        return tasks,prior

    def test_frozen_scale_and_prior_are_preserved(self):
        tasks,prior=self.fixture();before=repr(prior);out,check=fit_multi(tasks,prior)
        self.assertEqual(out['model']['mean'],prior['model']['mean'])
        self.assertEqual(out['model']['scale'],prior['model']['scale'])
        self.assertEqual(repr(prior),before)
        self.assertAlmostEqual(out['model']['coef'][2],.7,places=8)
        self.assertLess(check['KKT_stationarity'],1e-4)

    def test_informative_feedback_changes_decision(self):
        tasks,prior=self.fixture();out,_=fit_multi(tasks,prior)
        self.assertGreater(out['model']['coef'][1],0.)

    def test_all_flat_feedback_cannot_trigger_fit(self):
        tasks,prior=self.fixture()
        for t in tasks:t['gain']=0.
        with self.assertRaises(ValueError):fit_multi(tasks,prior)

    def test_state_weights_use_task_differences(self):
        tasks,prior=self.fixture()
        tasks+=[dict(t,state='t',gain=t['gain']*3) for t in tasks[:]]
        _,check=fit_multi(tasks,prior)
        np.testing.assert_allclose(check['group_weights'],[.25,.75])


if __name__=='__main__':unittest.main()
