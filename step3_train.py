"""
STEP 3 — Train CascadeEWS
"""
import numpy as np, torch, torch.nn as nn
import os, sys, time, json
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
sys.path.insert(0,".")
from src.model import CascadeEWS

print("="*60); print("STEP 3 — Training CascadeEWS"); print("="*60)

nf  = np.load("data/processed/node_features.npy")
ei  = np.load("data/processed/edge_index.npy")
ew  = np.load("data/processed/edge_weights.npy")
gs  = np.load("data/processed/global_score.npy")
mc  = np.load("data/processed/multiclass_label.npy")
nr  = np.load("data/processed/node_risk.npy")
grid= np.load("data/processed/grid_shape.npy")
with open("data/processed/metadata.json") as f: meta=json.load(f)

T,N,F = nf.shape
NLAT,NLON = int(grid[0]),int(grid[1])
print(f"\n  Data: T={T} N={N} F={F}")

# Adjacency
adj = np.zeros((N,N),dtype=np.float32)
for i in range(ei.shape[1]):
    adj[ei[0,i],ei[1,i]] = ew[i]
adj /= (adj.sum(1,keepdims=True)+1e-8)

# Dataset
SEQ=12
X,yr,yc,yk=[],[],[],[]
for t in range(SEQ,T):
    X.append(nf[t-SEQ:t].transpose(1,0,2))
    yr.append(gs[t]); yc.append(mc[t]); yk.append(nr[t])
X=np.array(X,dtype=np.float32)
yr=np.array(yr,dtype=np.float32)
yc=np.array(yc,dtype=np.int64)
yk=np.array(yk,dtype=np.float32)
ym,ys=yr.mean(),yr.std()+1e-8
yr_n=(yr-ym)/ys

tr=int(len(X)*0.70); vl=int(len(X)*0.85)
X_tr,X_vl,X_te   = X[:tr],X[tr:vl],X[vl:]
yr_tr,yr_vl,yr_te = yr_n[:tr],yr_n[tr:vl],yr_n[vl:]
yc_tr,yc_vl,yc_te = yc[:tr],yc[tr:vl],yc[vl:]
yk_tr,yk_vl,yk_te = yk[:tr],yk[tr:vl],yk[vl:]
print(f"  Train:{len(X_tr)} Val:{len(X_vl)} Test:{len(X_te)}")
print(f"  Classes: {dict(zip(*np.unique(yc_tr,return_counts=True)))}")
np.save("data/processed/reg_stats.npy",np.array([ym,ys]))
np.save("results/X_test.npy",X_te)
np.save("results/yr_test.npy",yr[vl:])
np.save("results/yc_test.npy",yc_te)
np.save("results/yk_test.npy",yk_te)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
ei_t=torch.LongTensor(ei); ew_t=torch.FloatTensor(ew)
adj_t=torch.FloatTensor(adj)

model=CascadeEWS(in_feat=F,hidden=64,seq_len=SEQ,
                 n_nodes=N,n_classes=3,dropout=0.2).to(device)
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

opt=AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
sch=CosineAnnealingLR(opt,T_max=50)
mse=nn.MSELoss()
ce =nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0,2.0,4.0]).to(device))

EPOCHS=30; BATCH=16; best=float("inf")
tl,vl_l=[],[]
os.makedirs("results/checkpoints",exist_ok=True)

print(f"\n  {'Ep':>4}|{'Train':>8}|{'Val':>8}|{'ETA':>7}")
print("  "+"-"*35)
t0=time.time()

for ep in range(EPOCHS):
    model.train()
    perm=np.random.permutation(len(X_tr))
    el=0.0; nb=0
    for b in range(0,len(perm),BATCH):
        idx=perm[b:b+BATCH]
        opt.zero_grad()
        total=torch.tensor(0.0,device=device)
        for i in idx:
            xi=torch.FloatTensor(X_tr[i]).to(device)
            ri=torch.FloatTensor([yr_tr[i]]).to(device)
            ci=torch.LongTensor([yc_tr[i]]).to(device)
            ki=torch.FloatTensor(yk_tr[i]).to(device)
            reg,clf,risk,_,_,_,phys=model(xi,ei_t,ew_t,adj_t)
            # Fix shapes
            l_reg =mse(reg.view(1),ri.view(1))
            l_clf =ce(clf.view(1,3),ci.view(1))
            ki_n  =(ki-ki.min())/(ki.max()-ki.min()+1e-8)
            l_risk=mse(risk,ki_n)
            loss  =l_reg+l_clf+0.3*l_risk+0.1*phys
            total =total+loss
        (total/len(idx)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); el+=total.item()/len(idx); nb+=1
    avg_tr=el/nb; tl.append(avg_tr)

    model.eval(); vl_loss=0.0
    with torch.no_grad():
        for i in range(len(X_vl)):
            xi=torch.FloatTensor(X_vl[i]).to(device)
            ri=torch.FloatTensor([yr_vl[i]]).to(device)
            ci=torch.LongTensor([yc_vl[i]]).to(device)
            ki=torch.FloatTensor(yk_vl[i]).to(device)
            reg,clf,risk,_,_,_,phys=model(xi,ei_t,ew_t,adj_t)
            l=mse(reg.view(1),ri.view(1))+ce(clf.view(1,3),ci.view(1))
            vl_loss+=l.item()
    avg_vl=vl_loss/len(X_vl); vl_l.append(avg_vl); sch.step()

    if avg_vl<best:
        best=avg_vl
        torch.save({"model_state":model.state_dict(),
                    "train_losses":tl,"val_losses":vl_l,
                    "config":{"in_feat":F,"hidden":64,"seq_len":SEQ,
                              "n_nodes":N,"nlat":NLAT,"nlon":NLON,
                              "n_classes":3,"dropout":0.2},
                    "split":{"tr":tr,"vl":vl},
                    "reg_stats":[float(ym),float(ys)],
                    "metadata":meta},
                   "results/checkpoints/cascadeews_best.pt")

    eta=(time.time()-t0)/(ep+1)*(EPOCHS-ep-1)/60
    if (ep+1)%5==0:
        tag=" ★" if avg_vl==best else ""
        print(f"  {ep+1:>4}|{avg_tr:>8.4f}|{avg_vl:>8.4f}|{eta:>5.1f}m{tag}")

print(f"\n✓ TRAINING COMPLETE | Best val: {best:.4f}")
print("Next → python step4_evaluate.py")
