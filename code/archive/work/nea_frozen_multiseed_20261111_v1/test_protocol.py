"""只验证扩充清单和统计口径，不生成线路。"""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from experiment import schedule
from summarize import point,stats


class Checks(unittest.TestCase):
    def test_schedule_complete_balanced(self):
        contexts=[{'parent':str(i)} for i in range(8)];methods=['author','full','count','ridge']
        tasks=schedule(contexts,list(range(2000,2010)),methods)
        self.assertEqual(len(tasks),320)
        self.assertEqual(len({(c['parent'],s,m) for c,s,m in tasks}),320)
        for method in methods:
            positions=[0]*4
            for i,(_,_,m) in enumerate(tasks):
                if m==method:positions[i%4]+=1
            self.assertEqual(positions,[20]*4)

    def test_cutoff_uses_only_completed_points(self):
        run={'fine_curve':[{'seconds':0.5,'cost':10.},{'seconds':4.,'cost':8.},{'seconds':4.01,'cost':1.}]}
        self.assertEqual(point(run,4.)['cost'],8.)

    def test_mean_ratio_preserves_losses(self):
        rows=[{'author_budget':1.,'count_budget':.5},{'author_budget':3.,'count_budget':4.}]
        s=stats(rows,'count')
        self.assertEqual(s['improvement_percent'],-12.5)
        self.assertEqual((s['wins'],s['ties'],s['losses']),(1,0,1))


if __name__=='__main__':unittest.main()
