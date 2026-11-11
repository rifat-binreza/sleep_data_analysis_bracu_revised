"""Stack-component ablation under the primary protocol (grouped 10x5 CV). usage: ablation_stack.py <rep_from> <rep_to>
All variants share identical outer splits (same as run_cv.py grouped), identical inner OOF probabilities and identical fitted base learners."""
import sys, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from common import *
R = "../results/"; a, b = int(sys.argv[1]), int(sys.argv[2])
X, y = load(); g = group_ids(X)
def inner(Xtr, ytr, seed):
    cats = vocab(Xtr); n = len(Xtr); sp = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xtr, ytr, group_ids(Xtr))
    Prf = np.zeros((n, 3)); Px = np.zeros((n, 3)); pas = None
    for tr, va in sp:
        pre = make_pre(cats).fit(Xtr.iloc[tr]); A, B = pre.transform(Xtr.iloc[tr]), pre.transform(Xtr.iloc[va])
        if pas is None: pas = np.zeros((n, B.shape[1]))
        Prf[va] = rf(seed).fit(A, ytr[tr]).predict_proba(B); Px[va] = xgb(seed).fit(A, ytr[tr]).predict_proba(B); pas[va] = B
    return cats, pas, Prf, Px
def task(rep, fold, tr, te):
    seed = 1000 + rep; Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True); ytr = y[tr]
    cats, pas, Prf, Px = inner(Xtr, ytr, seed); pre = make_pre(cats).fit(Xtr); A, B = pre.transform(Xtr), pre.transform(Xte)
    r = rf(seed).fit(A, ytr); x = xgb(seed).fit(A, ytr); pr, px = r.predict_proba(B), x.predict_proba(B); out = {}
    def meta(Ztr, Zte): return LogisticRegression(C=2.0, max_iter=5000).fit(Ztr, ytr).predict_proba(Zte)
    out["Full stack (RF+XGB+passthrough)"] = meta(np.hstack([pas, Prf, Px]), np.hstack([B, pr, px]))
    out["Stack without RF"] = meta(np.hstack([pas, Px]), np.hstack([B, px]))
    out["Stack without XGBoost"] = meta(np.hstack([pas, Prf]), np.hstack([B, pr]))
    out["Stack without passthrough"] = meta(np.hstack([Prf, Px]), np.hstack([pr, px]))
    out["Soft voting (RF+XGB)"] = (pr + px) / 2; out["RF only"] = pr; out["XGBoost only"] = px
    return rep, fold, te, out
jobs = []
for rep in range(a, b):
    for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(X, y, g)): jobs.append((rep, fold, tr, te))
res = Parallel(n_jobs=2)(delayed(task)(*j) for j in jobs); rows = []
for rep, fold, te, out in res:
    yt = y[te]
    for m, p in out.items():
        yh = p.argmax(1); rows.append(dict(repeat=rep, fold=fold, model=m, acc=accuracy_score(yt, yh), macro_f1=f1_score(yt, yh, average="macro", labels=[0,1,2]),
              auc=roc_auc_score(yt, p, multi_class="ovr", average="macro", labels=[0,1,2]), logloss=log_loss(yt, np.clip(p, 1e-12, 1), labels=[0,1,2])))
pd.DataFrame(rows).to_csv(R + f"ablation_stack_{a}_{b}.csv", index=False); print("done", a, b)
