import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import experiment as e


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pilot,cls.runtime=e.load_runtime()

    def test_both_methods_use_identical_online_allocation(self):
        full=e.make(self.pilot,self.runtime,'full')
        ridge=e.make(self.pilot,self.runtime,'ridge')
        self.assertEqual(full.__code__.co_code,ridge.__code__.co_code)
        self.assertTrue(full.__globals__['online'])
        self.assertTrue(ridge.__globals__['online'])
        self.assertIs(full.__globals__['BatchAllocator'],ridge.__globals__['BatchAllocator'])
        models=e.read(e.WORK/'nea_paper_final_confirm_20261110_v1/模型冻结.json')
        self.assertEqual(full.__globals__['frozen']('mumford0'),models['full']['mumford0'])
        self.assertEqual(ridge.__globals__['frozen']('mumford0'),models['ridge']['mumford0'])
        self.assertEqual(full.__globals__['frozen']('mumford0'),models['full']['mumford0'])

    def test_frozen_tasks_no_author_or_count_searches(self):
        req=e.read(e.REQUEST)
        self.assertEqual(len(req['tasks']),80)
        self.assertEqual({v for _,_,v in req['tasks']},{'full','ridge'})
        self.assertEqual({c['city'] for c,_,_ in req['tasks']},{'mumford0'})
        self.assertEqual(len({(c['parent'],s,v) for c,s,v in req['tasks']}),80)
        e.check(req['hashes'])


if __name__=='__main__':
    unittest.main()
