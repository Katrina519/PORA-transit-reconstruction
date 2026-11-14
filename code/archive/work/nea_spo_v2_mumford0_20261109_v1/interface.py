"""第2版的实例接口；作者实现及冻结评分由原文件提供。"""
import sys
import inspect
import copy
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
sys.path.insert(0, str(WORK / 'nea_spo_cost_scale_20261108_v1'))
import run_stage as stage

q, r, np, torch = stage.q, stage.r, stage.q.np, stage.torch
DLN = WORK / 'paper_code_learning_20261030_v1/dln/author_source/Codes and Instances/Networks and Line plans'
MODEL_PATH = WORK / 'nea_spo_v2_matched_ablation_20261108_v1/模型冻结.json'
MODEL = stage.read(MODEL_PATH)['models']['v2_spo']
ORIGINAL_COLONY = r.entry.bee_colony
SOURCES = {name: inspect.getsource(fn) for name, fn in [
    ('build', q.a.build), ('assess', q.assess), ('selection', stage.BASE_SELECTION),
    ('timed_one', stage.timed_one)]}
MANDL_ROADS, MANDL_Q = q.ROADS.copy(), q.Q.copy()
REUSE = []
SPEC = {}
BASE_SELECTION = None


def compile_function(name, changes, namespace):
    text = SOURCES[name]
    for old, new, expected in changes:
        assert text.count(old) == expected, (name, old, text.count(old))
        text = text.replace(old, new)
    scope = dict(namespace)
    scope['SPEC'] = SPEC
    exec(compile(text, str(ROOT / 'interface.py') + '::' + name, 'exec'), scope)
    REUSE.append({'function': name, 'replacements': [list(v) for v in changes]})
    return scope[name]


def configure(city='mumford0'):
    global SPEC, BASE_SELECTION
    REUSE.clear()
    if city == 'mumford0':
        SPEC = {'city': 'Mumford0', 'n': 30, 'routes': 12, 'min': 2, 'max': 15, 'diameter': 26.}
        q.ROADS = np.loadtxt(DLN / 'Mumford0Network.txt')
        q.Q = np.loadtxt(DLN / 'Mumford0Demand.txt')
        input_dir = ROOT / 'inputs'
    else:
        assert city == 'mandl'
        SPEC = {'city': 'Mandl', 'n': 15, 'routes': 6, 'min': 2, 'max': 8, 'diameter': 33.}
        q.ROADS, q.Q = MANDL_ROADS.copy(), MANDL_Q.copy()
        input_dir = q.a.INPUTS
    q.MODEL = copy.deepcopy(MODEL)
    q.a.build = compile_function('build', [
        ("'city':'Mandl'", "'city':SPEC['city']", 1),
        ('RouteGenBatchState(data,cost,6,2,8)',
         "RouteGenBatchState(data,cost,SPEC['routes'],SPEC['min'],SPEC['max'])", 1)],
        {**q.a.__dict__, 'INPUTS': input_dir})
    q.assess = compile_function('assess', [
        ('len(routes)==6', "len(routes)==SPEC['routes']", 1),
        ('set(range(15))', "set(range(SPEC['n']))", 1),
        ('2<=len(rr)<=8', "SPEC['min']<=len(rr)<=SPEC['max']", 1)], q.__dict__)
    BASE_SELECTION = compile_function('selection', [
        ('range(15)', "range(SPEC['n'])", 1),
        ('range(6)', "range(SPEC['routes'])", 3),
        ('j+1:],15,pieces)', "j+1:],SPEC['n'],pieces)", 1)], q.__dict__)
    q.selection = BASE_SELECTION


def make_one():
    r.entry.bee_colony = ORIGINAL_COLONY
    factory = compile_function('timed_one', [
        ('source=inspect.getsource(r.one)',
         "source=inspect.getsource(r.one).replace('max_route_len=8', 'max_route_len=' + str(SPEC['max']))", 1)], stage.__dict__)
    return factory()


def run_one(one, context, seed, method, iterations=8):
    started = time.perf_counter()
    cache, counts = {}, [0, 0]
    def select(routes, model, state, base, probe):
        key = tuple(tuple(rr) for rr in routes)
        if key not in cache:
            result = dict(BASE_SELECTION(routes, model, state, base, probe))
            result['allowed'] = list(range(SPEC['routes']))
            result['ranking'] = sorted(result['allowed'], key=lambda j: (-result['scores'][j], j))
            result['chosen'] = result['ranking'][0]
            cache[key] = result
            counts[0] += 1
        else:
            counts[1] += 1
        return cache[key]
    q.selection = select
    offset = time.perf_counter() - started
    item = one(context, seed, 'selector' if method == 'v2_spo' else method, iterations)
    item.update(method=method, score_calls=counts[0], cache_hits=counts[1])
    for point in item['fine_curve']:
        point['seconds'] += offset
    for solution in item['fine_solutions']:
        solution['first_seconds'] += offset
    item['fine_checkpoints'] = {str(b): stage.at_budget(item['fine_curve'], b) for b in [2., 4.]}
    for point in item['curve']:
        point['seconds'] += offset
    item['checkpoints'] = {str(b): next((v['cost'] for v in item['curve'][::-1] if v['seconds'] <= b), None)
                           for b in [1., 2., 4.]}
    item['initialized_at_checkpoint'] = {str(b): item['curve'][0]['seconds'] <= b for b in [1., 2., 4.]}
    item['seconds'] = time.perf_counter() - started
    item['wrapper_preparation_seconds'] = offset
    return item


def reference_routes():
    return [[int(v)-1 for v in line.split()] for line in
            (DLN / 'Mumford0Lineplan.txt').read_text().splitlines() if line.strip()]
