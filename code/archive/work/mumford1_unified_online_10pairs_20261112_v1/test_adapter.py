import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import experiment as e


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.pilot,cls.runtime=e.load_runtime()

    def test_configuration_and_frozen_fifteen_route_scores(self):
        import numpy as np
        runtime=self.runtime
        doc=runtime.d.frozen('mumford1')
        before=e.sha(e.WORK/'nea_spo_v2_joint_20261109_v1/模型冻结.json')
        self.pilot.configure(runtime,doc)
        self.assertEqual(runtime.d.b.base.SPEC,self.pilot.SPEC)
        self.assertEqual(runtime.d.q.ROADS.shape,(70,70))
        x=np.arange(150,dtype=float).reshape(15,10)/100.
        np.testing.assert_allclose(runtime.d.b.learner.scores(doc,x,'mumford0'),runtime.d.q.ref.selector.scores(doc['model'],x),rtol=0,atol=1e-12)
        self.assertEqual(e.sha(e.WORK/'nea_spo_v2_joint_20261109_v1/模型冻结.json'),before)
        self.assertTrue(e.make(self.pilot,runtime,'full').__globals__['online'])
        self.assertFalse(e.make(self.pilot,runtime,'author').__globals__['online'])

    def test_feedback_index_and_parent_isolation_for_fifteen_tasks(self):
        alloc=self.runtime.d.BatchAllocator(.05,repair=False)
        scores=[2.]+[0.]*14
        a,b=((0,1),(1,2)),((0,2),(2,1))
        self.assertEqual(alloc.choose(a,scores,7,1.),0)
        self.assertEqual(alloc.states[a]['completed'],[0]*15)
        costs=[1.]*10;costs[7]=.8;valid=[False]*10;valid[7]=True
        alloc.observe(costs,valid)
        self.assertAlmostEqual(alloc.states[a]['sum'][0],.2)
        self.assertEqual(alloc.choose(b,scores,2,1.),0)
        self.assertEqual(alloc.states[b]['completed'],[0]*15)
        alloc.observe([1.]*10,[False]*10)
        self.assertEqual(alloc.states[b]['sum'],[0.]*15)

    def test_frozen_tasks_and_objective(self):
        req=e.read(ROOT/'请求冻结.json')
        self.assertEqual(len(req['tasks']),20)
        self.assertEqual(req['seeds'],list(range(3011,3021)))
        self.assertEqual(len({(s,v) for _,s,v in req['tasks']}),20)
        doc=self.runtime.d.frozen('mumford1');self.pilot.configure(self.runtime,doc)
        q=self.runtime.d.q
        value=q.assess(req['context']['routes'])
        self.assertTrue(value['feasible'])
        self.assertAlmostEqual(value['P']/88.+value['O']/660.,3.36954665,delta=2e-7)
        e.check(req['hashes'])


if __name__=='__main__':unittest.main()
