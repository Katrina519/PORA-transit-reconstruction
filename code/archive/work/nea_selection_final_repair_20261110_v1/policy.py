"""精确父网内的边际反馈；不读取种子、案例编号或未来结果。"""
from allocator import Allocator


class MarginalAllocator(Allocator):
    def __init__(self, *args, repair=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.best = {}

    def observe(self, costs, valid):
        for key, j, bee, base, index in self.pending:
            gain = max(0., base-float(costs[bee])) if valid[bee] else 0.
            if gain < 1e-6:
                gain = 0.
            old = self.best.get((key, j), 0.)
            reward = max(0., gain-old)/base
            self.best[key, j] = max(old, gain)
            state = self.states[key]
            state['completed'][j] += 1
            state['sum'][j] += reward
            self.log[index].update(reward=reward, valid=bool(valid[bee]), raw_gain=gain)
        self.pending = []


class RetryOrder:
    def __init__(self):
        self.records = {}

    def choose(self, key, scores):
        records = self.records.setdefault(key, [[0., 0] for _ in scores])
        return min(range(len(scores)), key=lambda j: (records[j][1]//2, -scores[j], j))

    def observe(self, events):
        for event in events:
            key = tuple(tuple(r) for r in event['parent_routes'])
            record = self.records[key][event['j']]
            gain = max(0., event['parent_cost']-event['candidate_cost']) if event['valid'] else 0.
            if gain > record[0]+1e-6:
                record[:] = [gain, 0]
            else:
                record[1] += 1
