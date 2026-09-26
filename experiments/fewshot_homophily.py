from __future__ import annotations
import argparse, sys, csv
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data import DATASETS, load_dataset, fewshot_split, set_seed
from src.graph_ops import lazy_random_walk
from src.odinn import build_odinn
from src.training import train_odinn
from src.io_utils import append_csv, completed_keys, log_failure

FIELDS=['dataset','model','shots','hops','seed','val_acc','test_acc','best_epoch','train_seconds','status']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--datasets',nargs='*',default=DATASETS); ap.add_argument('--models',nargs='*',default=['dg','fj']); ap.add_argument('--shots',nargs='*',type=int,default=[1,3,5]); ap.add_argument('--hops',nargs='*',type=int,default=[10,20,40]); ap.add_argument('--folds',type=int,default=30); ap.add_argument('--epochs',type=int,default=1000); ap.add_argument('--patience',type=int,default=50); ap.add_argument('--lr',type=float,default=0.005); ap.add_argument('--weight_decay',type=float,default=5e-4); ap.add_argument('--dropout',type=float,default=0.6); ap.add_argument('--hidden',type=int,default=64); ap.add_argument('--root',default='data'); ap.add_argument('--out_dir',default='results/fewshot_homophily'); ap.add_argument('--cpu',action='store_true'); args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); runs=out/'odinn_all_runs.csv'; fail=out/'failures.csv'; done=completed_keys(runs,['dataset','model','shots','hops','seed'])
    device=torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    for dsname in args.datasets:
        try:
            bundle=load_dataset(dsname,args.root); data=bundle.data; w=lazy_random_walk(data.edge_index,data.num_nodes,device=device)
        except Exception as e:
            log_failure(fail,f'load:{dsname}',e); continue
        for model_name in args.models:
            for shots in args.shots:
                for hops in args.hops:
                    for fold in range(args.folds):
                        seed=42+fold; key=(bundle.name,model_name,str(shots),str(hops),str(seed))
                        if key in done: continue
                        task=f'{bundle.name}/{model_name}/shot{shots}/K{hops}/seed{seed}'
                        try:
                            set_seed(seed); masks=fewshot_split(data.y,shots,seed); model=build_odinn(model_name,data.num_features,bundle.num_classes,hops,args.hidden,args.dropout)
                            res,_=train_odinn(model,data,masks,w,device,args.epochs,args.lr,args.weight_decay,args.patience); res.seed=seed
                            row={'dataset':bundle.name,'model':model_name,'shots':shots,'hops':hops,'seed':seed,'val_acc':res.val_acc,'test_acc':res.test_acc,'best_epoch':res.best_epoch,'train_seconds':res.train_seconds,'status':'ok'}; append_csv(runs,row,FIELDS); done.add(key)
                        except RuntimeError as e:
                            status='oom' if 'out of memory' in str(e).lower() else 'error'; append_csv(runs,{'dataset':bundle.name,'model':model_name,'shots':shots,'hops':hops,'seed':seed,'val_acc':'','test_acc':'','best_epoch':'','train_seconds':'','status':status},FIELDS); log_failure(fail,task,e); 
                            if torch.cuda.is_available(): torch.cuda.empty_cache()
                        except Exception as e: append_csv(runs,{'dataset':bundle.name,'model':model_name,'shots':shots,'hops':hops,'seed':seed,'val_acc':'','test_acc':'','best_epoch':'','train_seconds':'','status':'error'},FIELDS); log_failure(fail,task,e)
    if runs.exists():
        df=pd.read_csv(runs); df=df[df.status=='ok'].copy();
        if len(df):
            agg=df.groupby(['dataset','model','shots','hops']).agg(val_mean=('val_acc','mean'),test_mean=('test_acc','mean'),test_std=('test_acc',lambda x: x.std(ddof=0)),n=('test_acc','count')).reset_index()
            eligible=agg[agg.n>=args.folds].copy()
            best=(eligible.sort_values(['dataset','model','shots','val_mean'],ascending=[True,True,True,False])
                  .groupby(['dataset','model','shots'],as_index=False).first())
            best.to_csv(out/'odinn_best_summary.csv',index=False); agg.to_csv(out/'odinn_hop_summary.csv',index=False)
            print(best.to_string(index=False))
if __name__=='__main__': main()
