"""Run FT-Transformer under the identical outer splits as run_cv.py and append its rows to folds_/oof_ files.  usage: run_extra.py <proto> <repeats>"""
import sys, time, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, balanced_accuracy_score
from common import *
from ftt import FTTransformer
proto, Rr = sys.argv[1], int(sys.argv[2]); K = 5; R = "../results/"
X, y = load(); g = group_ids(X)
if proto == "dedup":
    keep = ~X.assign(y=y).duplicated().values; X, y = X[keep].reset_index(drop=True), y[keep]; g = group_ids(X)
def task(rep, fold, tr, te):
    m = FTTransformer(NUM, CAT, seed=1000 + rep).fit(X.iloc[tr].reset_index(drop=True), y[tr]); return rep, fold, te, m.predict_proba(X.iloc[te].reset_index(drop=True))
jobs = []
for rep in range(Rr):
    sp = StratifiedGroupKFold(K, shuffle=True, random_state=rep).split(X, y, g) if proto == "grouped" else StratifiedKFold(K, shuffle=True, random_state=rep).split(X, y)
    for fold, (tr, te) in enumerate(sp): jobs.append((rep, fold, tr, te))
t0 = time.time(); res = Parallel(n_jobs=2)(delayed(task)(*j) for j in jobs); print("secs", round(time.time() - t0))
rows, preds = [], []; M = "FT-Transformer"
for rep, fold, te, p in res:
    yt = y[te]; yh = p.argmax(1); Y = np.eye(3)[yt]
    rows.append(dict(protocol=proto, repeat=rep, fold=fold, model=M, n_test=len(te), acc=accuracy_score(yt, yh), macro_f1=f1_score(yt, yh, average="macro", labels=[0,1,2]),
        bal_acc=balanced_accuracy_score(yt, yh), logloss=log_loss(yt, np.clip(p, 1e-12, 1), labels=[0,1,2]), brier=float(((p - Y) ** 2).sum(1).mean()),
        auc=roc_auc_score(yt, p, multi_class="ovr", average="macro", labels=[0,1,2])))
    for i, idx in enumerate(te): preds.append(dict(protocol=proto, repeat=rep, fold=fold, model=M, row=int(idx), y=int(y[idx]), pred=int(yh[i]), p_insomnia=p[i,0], p_none=p[i,1], p_apnea=p[i,2]))
f0 = pd.read_csv(R + f"folds_{proto}.csv"); f0 = f0[f0.model != M]; pd.concat([f0, pd.DataFrame(rows)]).to_csv(R + f"folds_{proto}.csv", index=False)
o0 = pd.read_csv(R + f"oof_predictions_{proto}.csv"); o0 = o0[o0.model != M]; pd.concat([o0, pd.DataFrame(preds)]).to_csv(R + f"oof_predictions_{proto}.csv", index=False)
d = pd.DataFrame(rows); print(d[["acc", "macro_f1", "brier", "logloss", "auc"]].agg(["mean", "std"]).round(3))
