"""原作者固定推理策略的只读概率适配，不修改作者源文件。"""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SRC=Path(r'C:\Users\m\Desktop\work\paper_code_learning_20261030_v1\holliday\source_fixed')
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(ROOT/'mplconfig'),WANDB_MODE='disabled')
sys.path[:0]=[str(SRC),str(SRC/'learning')]
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from omegaconf import OmegaConf
from hydra import initialize_config_dir,compose
from simulation.citygraph_dataset import get_dataset_from_config
from simulation.transit_time_estimator import RouteGenBatchState,get_cost_module_from_cfg
from learning.utils import build_model_from_cfg
from learning import models
import torch_utils as tu
from unittest.mock import patch

WEIGHT=Path(r'C:\Users\m\Desktop\work\开源公交复现\ea_nea_mandl_smoke_v1\inputs\inductive_gae_seed_0.pt')
INPUTS=SRC.parent/'inputs'

def route(state):
    return tuple(int(v) for v in state.current_routes[0].tolist() if v>=0)

def build(kind,remaining):
    with initialize_config_dir(version_base=None,config_dir=str(SRC/'cfg')):
        cfg=compose(config_name='neural_bco_mumford',overrides=['model=random_path_combiner'] if kind=='random' else [])
    cfg.experiment.cost_function.kwargs.variable_weights=False
    cfg.experiment.cost_function.kwargs.demand_time_weight=.5
    cfg.experiment.cost_function.kwargs.route_time_weight=.5
    model=build_model_from_cfg(cfg.model,cfg.experiment)
    if kind=='neural':model.load_state_dict(torch.load(WEIGHT,map_location='cpu'))
    model.eval();model.freeze_feature_norms()
    cost=get_cost_module_from_cfg(cfg.experiment.cost_function,False,True)
    data=get_dataset_from_config(OmegaConf.create({'type':'mumford','path':str(INPUTS),'city':'Mandl'}))[0]
    state=RouteGenBatchState(data,cost,6,2,8)
    state.add_new_routes(tu.get_batch_tensor_from_routes([remaining]))
    model.setup_planning(state)
    return model,state

def distribution(model,state):
    captured=[]
    original=models.select
    def capture(scores,scores_as_logits=False,softmax_temp=1,n_selections=1,selection=None):
        lp=models.logsoftmax(scores,softmax_temp).to(torch.float64)
        captured.append(lp.detach().clone())
        # 首次halt选择：有非零继续概率时选择继续，确保第二步为延伸分布。
        # 被强制停止时选择停止；第二步按原分布最大值选合法动作。
        chosen=lp.argmax(-1)
        if len(captured)==1 and float(lp[0,0].exp())>0:
            chosen=torch.zeros_like(chosen)
        return original(scores,scores_as_logits,softmax_temp,n_selections,selection=chosen)
    with torch.no_grad(),patch.object(models,'select',capture):
        model.step(state,greedy=False)
    assert model.serial_halting and len(captured)==2
    corh=captured[0].exp()[0]
    ext=captured[1].exp()[0]
    sequences=state.get_shortest_path_sequences()[0]
    current=route(state)
    out=[]
    for index,p in enumerate(ext.tolist()):
        prob=float(corh[0])*p
        if prob==0:continue
        a,b=divmod(index,int(state.max_n_nodes))
        seq=tuple(int(v) for v in sequences[a,b].tolist() if v>=0)
        if not current:nxt=seq
        elif b==current[0]:nxt=seq+current[1:]
        elif a==current[-1]:nxt=current+seq[1:]
        else:raise AssertionError(('non-end extension',current,a,b))
        assert len(nxt)>len(current) and len(set(nxt))==len(nxt)
        out.append(((a,b),nxt,prob))
    halt=float(corh[1])
    assert abs(halt+sum(p for _,_,p in out)-1)<1e-5
    return halt,out

def apply(state,action):
    new=state.clone()
    new.shortest_path_action(torch.tensor([action],dtype=torch.long))
    return new

def state_difference(a,b):
    keys=('current_routes','route_mat','transit_times','n_transfers','has_path','current_route_time','current_route_times_from_start','valid_terms_mat','total_route_time')
    maxdiff=0.
    for k in keys:
        x,y=getattr(a,k),getattr(b,k)
        assert torch.equal(torch.isfinite(x),torch.isfinite(y)),k
        mask=torch.isfinite(x)
        if mask.any():
            err=float((x[mask].double()-y[mask].double()).abs().max())
            maxdiff=max(maxdiff,err)
    maxdiff=max(maxdiff,float((a.get_global_state_features()-b.get_global_state_features()).abs().max()))
    return maxdiff

def query(model,root,target):
    """只访问目标内可达区间；新对象查询不使用全图或全对象质量。"""
    L=len(target);prob={};representatives={};calls=0
    h0,rootedges=distribution(model,root);calls+=1
    intervals={target[i:j] for i in range(L) for j in range(i+2,L+1)}
    for action,nxt,p in rootedges:
        if nxt in intervals:
            prob[nxt]=prob.get(nxt,0.)+p
            representatives.setdefault(nxt,apply(root,action))
    answer=0.
    for length in range(2,L+1):
        for i in range(L-length+1):
            p=target[i:i+length]
            if p not in prob:continue
            state=representatives[p]
            halt,edges=distribution(model,state);calls+=1
            if p==target:answer+=prob[p]*halt
            for action,nxt,pe in edges:
                if nxt in intervals:
                    prob[nxt]=prob.get(nxt,0.)+prob[p]*pe
                    representatives.setdefault(nxt,apply(state,action))
    return answer,calls
