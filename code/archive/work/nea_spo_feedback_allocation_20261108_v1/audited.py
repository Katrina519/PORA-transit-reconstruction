"""仅补证据字段，不改变任务分配决定。"""
from allocator import Allocator
class AuditedAllocator(Allocator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.state_ids={};self.batch_id=0
    def choose(self,key,scores,bee,base):
        j=super().choose(key,scores,bee,base)
        if self.enabled:
            if key not in self.state_ids:self.state_ids[key]=len(self.state_ids)
            self.log[-1].update(state_id=self.state_ids[key],parent_routes=[list(rr) for rr in key],
                                parent_cost=base,batch_id=self.batch_id,prior=self.states[key]['prior'][:])
        return j
    def observe(self,costs,valid):
        for key,j,bee,base,index in self.pending:self.log[index]['candidate_cost']=float(costs[bee])
        super().observe(costs,valid);self.batch_id+=1
