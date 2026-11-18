"""Confidence calibration: is the model's stated confidence trustworthy?

This drives the app's "ask the user to sign again" behaviour, so it needs to be
measured rather than assumed. Reports an accuracy table by confidence band and
the accuracy obtained by abstaining below a threshold.

Run on the native cache (the geometry the app uses):
    python scripts/ml/confidence_calibration.py
"""
import sys, numpy as np, torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'/'ml'))
import robustness as R
R.CACHE_ROOT = ROOT / '.scratch' / 'landmarks_native'
from train import TemporalModel
from features import build_features

CK=[".scratch/runs/tcn21.pt",".scratch/runs/tcnm61.pt",".scratch/runs/tcnm62.pt",
    ".scratch/runs/rob71.pt"]+[f".scratch/search/mirror_handdrop_tcn_z_s{s}.pt" for s in (0,1,2,7,13)]
models=[];labels=None;frames=None
for c in CK:
    ck=torch.load(c,map_location='cpu',weights_only=True)
    if labels is None: labels=ck["vocab"]
    frames=int(ck["frames"])
    m=TemporalModel(len(ck["vocab"]),dim=int(ck["dim"]),frames=frames,
                    backbone=str(ck["backbone"]),in_dim=int(ck["model"]["stem.weight"].shape[1]))
    m.load_state_dict(ck["model"]); m.eval(); models.append(m)
idx={g:i for i,g in enumerate(labels)}

rows=[]
for p,lm,pres,l,_s in R.load_split('test',idx):
    L=[]
    for m in models:
        uz=m.stem.weight.shape[1]!=255
        f,mk=build_features(lm,pres,expected_frames=frames,use_z=uz)
        with torch.no_grad(): L.append(m(torch.from_numpy(f)[None],torch.from_numpy(mk)[None])[0].numpy())
    a=np.mean(L,axis=0); e=np.exp(a-a.max()); pr=e/e.sum()
    rows.append((float(pr.max()), int(pr.argmax())==l))
rows=np.array(rows, dtype=[('c','f8'),('ok','bool')])
bins=[(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.01)]
print(f"{'confidence':>14s} {'clips':>6s} {'accuracy':>9s} {'share of errors':>16s}")
E=rows[~rows['ok']].size
for lo,hi in bins:
    sel=(rows['c']>=lo)&(rows['c']<hi)
    if sel.sum()==0: continue
    acc=100*rows['ok'][sel].mean()
    errshare=100*(~rows['ok'][sel]).sum()/max(1,E)
    print(f'  {lo:.1f}-{hi:.1f}      {sel.sum():6d} {acc:8.1f}% {errshare:15.1f}%')
for t in (0.5,0.7):
    sel=rows['c']>=t
    print(f'  abstain below {t}: answers {100*sel.mean():.1f}% of clips at {100*rows["ok"][sel].mean():.2f}% accuracy')
