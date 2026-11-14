"""同批同父网的名额预约；关闭时沿用冻结算法。"""
from allocator import Allocator
import inspect


def available(n, reserved):
    return [j for j in range(n) if j not in reserved] or list(range(n))


class BatchAllocator(Allocator):
    def __init__(self, *args, repair=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.repair = repair


# 唯一算法差异是argmax的临时允许集合；不改先验、Q或反馈提交。
source = inspect.getsource(Allocator.choose)
source = source.replace('    def choose(', 'def choose(', 1)
lines = source.splitlines()
source = '\n'.join([lines[0]] + [line[4:] if line.startswith('    ') else line for line in lines[1:]])
old = 'j=min(range(len(scores)),key=lambda k:(-values[k],k))'
new = ('reserved={j for parent,j,bee,base,index in self.pending if parent==key} if self.repair else set()\n'
       '    j=min(available(len(scores),reserved),key=lambda k:(-values[k],k))')
assert source.count(old) == 1
namespace = dict(Allocator.choose.__globals__, available=available)
exec(compile(source.replace(old, new), __file__+'::batch_choose', 'exec'), namespace)
BatchAllocator.choose = namespace['choose']
