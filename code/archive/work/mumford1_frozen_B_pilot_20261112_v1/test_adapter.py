"""只验证接口和目标，不运行线路生成或搜索。"""
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pilot


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = pilot.load_runtime()
        cls.document = pilot.read(pilot.WORK / 'nea_paper_final_confirm_20261110_v1/模型冻结.json')['full']['mumford0']
        cls.model_before = pilot.sha(pilot.WORK / 'nea_spo_v2_joint_20261109_v1/模型冻结.json')

    def test_mumford1_configuration_and_frozen_scores(self):
        runtime = self.runtime
        pilot.configure(runtime, self.document)
        self.assertEqual(runtime.d.b.base.SPEC, pilot.SPEC)
        self.assertEqual(runtime.d.q.ROADS.shape, (70, 70))
        self.assertEqual(runtime.d.q.Q.shape, (70, 70))
        self.assertEqual(self.document['encoding'], 'raw')
        # 任意有限输入上的预测须保持冻结线性模型完全一致。
        x = np.arange(150, dtype=float).reshape(15, 10) / 100.
        original = runtime.d.b.learner.scores(self.document, x, 'mumford0')
        reference = runtime.d.q.ref.selector.scores(self.document['model'], x)
        np.testing.assert_allclose(original, reference, rtol=0., atol=1e-12)
        self.assertEqual(len(original), 15)
        self.assertEqual(self.model_before, pilot.sha(pilot.WORK / 'nea_spo_v2_joint_20261109_v1/模型冻结.json'))

    def test_reference_state_and_author_objective(self):
        runtime = self.runtime
        pilot.configure(runtime, self.document)
        routes = [[int(v)-1 for v in line.split()] for line in
                  (pilot.DATA / 'Mumford1Lineplan.txt').read_text().splitlines() if line.strip()]
        q, torch = runtime.d.q, runtime.d.torch
        torch.manual_seed(0)
        with torch.no_grad():
            model, state = q.a.build('neural', routes)
            value = runtime.d.r.cost_module()(state)
        independent = q.assess(routes)
        self.assertTrue(independent['feasible'])
        self.assertEqual(int(state.n_routes_to_plan[0]), 15)
        self.assertEqual(int(state.min_route_len[0]), 10)
        self.assertEqual(int(state.max_route_len[0]), 30)
        self.assertEqual(int(state.max_n_nodes), 70)
        expected = independent['P']/88. + independent['O']/660.
        self.assertAlmostEqual(float(value.cost[0]), expected, delta=2e-6)
        self.assertEqual(float(value.get_metrics()['# disconnected node pairs'][0]), 0.)
        self.assertEqual(float(value.get_metrics()['# stops out of bounds'][0]), 0.)
        self.assertEqual(self.document, pilot.read(pilot.WORK / 'nea_paper_final_confirm_20261110_v1/模型冻结.json')['full']['mumford0'])


if __name__ == '__main__':
    unittest.main()
