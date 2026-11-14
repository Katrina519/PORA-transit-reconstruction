"""同父网内共享已完成任务反馈；不读取未来结果。"""
import math
class Allocator:
    def __init__(self,c=.05,enabled=True,use_prior=True,use_feedback=True):
        self.c=c;self.enabled=enabled;self.use_prior=use_prior;self.use_feedback=use_feedback
        self.states={};self.pending=[];self.log=[]
    def choose(self,key,scores,bee,base):
        if not self.enabled:return min(range(len(scores)),key=lambda j:(-scores[j],j))
        if key not in self.states:
            n=len(scores);mx=max(scores)
            ps=[math.exp(max(-60.,s-mx)) if self.use_prior else 1. for s in scores];total=sum(ps)
            self.states[key]={'prior':[p/total for p in ps],'visits':[0]*n,'completed':[0]*n,'sum':[0.]*n}
        st=self.states[key];total=max(1,sum(st['visits']))
        values=[(st['sum'][j]/st['completed'][j] if st['completed'][j] and self.use_feedback else 0.)+
                self.c*st['prior'][j]*math.sqrt(total)/(1+st['visits'][j]) for j in range(len(scores))]
        j=min(range(len(scores)),key=lambda k:(-values[k],k))
        self.log.append({'j':j,'bee':bee,'visits_before':st['visits'][:],'completed_before':st['completed'][:],'index':values,'reward':None})
        st['visits'][j]+=1;self.pending.append((key,j,bee,base,len(self.log)-1));return j
    def observe(self,costs,valid):
        for key,j,bee,base,log_index in self.pending:
            gain=max(0.,base-float(costs[bee])) if valid[bee] else 0.
            if gain<1e-6:gain=0.
            reward=gain/base
            assert 0.<=reward<=1.+1e-6
            st=self.states[key];st['completed'][j]+=1;st['sum'][j]+=reward
            self.log[log_index].update(reward=reward,valid=bool(valid[bee]))
        self.pending=[]
