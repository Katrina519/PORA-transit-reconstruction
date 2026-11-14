"""捕获六线硬编码拒绝合法十二线、或忽略新增线路的接口错误。"""
import sys
import unittest
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import interface as app


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.configure()
        cls.routes = app.reference_routes()

    def test_legal_twelve_line_network_is_not_rejected(self):
        self.assertTrue(app.q.assess(self.routes)['feasible'])

    def test_eleven_lines_are_not_accepted_as_twelve(self):
        self.assertFalse(app.q.assess(self.routes[:-1])['feasible'])

    def test_repeated_stop_is_not_accepted(self):
        bad = [rr[:] for rr in self.routes]
        bad[-1] = bad[-1][:3] + [bad[-1][1], bad[-1][0]]
        self.assertFalse(app.q.assess(bad)['feasible'])

    def test_sixteen_distinct_stops_exceed_route_limit(self):
        bad = [rr[:] for rr in self.routes]
        for j, rr in enumerate(bad):
            extensions = [u for u in range(30) if u not in rr and 0 < app.q.ROADS[rr[-1], u] < app.np.inf]
            if len(rr) == 15 and extensions:
                bad[j] = rr + [extensions[0]]
                self.assertFalse(app.q.assess(bad)['feasible'])
                return
        self.fail('reference fixture has no sixteen-stop simple extension')

    def test_all_stops_covered_but_disconnected_network_is_rejected(self):
        group = set(range(30)) - {0, 12}
        uncovered = set(group)
        routes = [[0, 12]]
        while uncovered:
            rr = [min(uncovered)]
            while len(rr) < 15:
                neighbors = [u for u in sorted(group) if u not in rr and 0 < app.q.ROADS[rr[-1], u] < app.np.inf]
                if not neighbors:
                    break
                rr.append(min(neighbors, key=lambda u: (u not in uncovered, u)))
            self.assertGreaterEqual(len(rr), 2)
            routes.append(rr); uncovered.difference_update(rr)
        self.assertLessEqual(len(routes), 12)
        routes += [routes[-1][:] for _ in range(12-len(routes))]
        self.assertEqual(set().union(*map(set, routes)), set(range(30)))
        self.assertFalse(app.q.assess(routes)['feasible'])

    def test_verifier_is_local_despite_imported_legacy_paths(self):
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).parent / 'run.py'),
                                 '--phase', 'check_verifier'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('LOCAL_VERIFIER_OK', result.stdout)


if __name__ == '__main__':
    unittest.main()
