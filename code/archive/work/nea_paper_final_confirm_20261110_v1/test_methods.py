import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from driver import CountPolicy
import importlib.util
spec=importlib.util.spec_from_file_location('paper_prepare_test',ROOT/'prepare.py')
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
fit_ridge=prep.fit_ridge

class Tests(unittest.TestCase):
    def test_count_uniform_and_isolated(self):
        p=CountPolicy();key=((0,1),)
        self.assertEqual([p.choose(key,3) for _ in range(8)],[0,1,2,0,1,2,0,1])
        self.assertEqual(p.choose(((1,0),),3),0)

    def test_ridge_is_deterministic(self):
        tasks=[{'city':'mumford0','state':'a','x':[float(j)]*10,'gain':float(j)} for j in range(3)]
        a,c=fit_ridge(tasks,'raw',.1);b,_=fit_ridge(tasks,'raw',.1)
        self.assertEqual(a,b);self.assertLess(c['normal_equation_residual'],1e-8)

if __name__=='__main__':unittest.main()
