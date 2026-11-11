"""Repeated nested CV: outer 5-fold (grouped or stratified-on-unique), inner 3-fold grouped selection of one
key hyper-parameter per model. All tuning touches the outer-training partition only.
usage: nested_cv.py <grouped|dedup> <repeats>"""
import sys, time, itertools, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, log_loss
from common import *

proto, R = sys.argv[1], int(sys.argv[2])
X, y = load()
if proto == "dedup":
    keep = ~X.assign(y=y).duplicated().values; X, y = X[keep].reset_index(drop=True), y[keep]
g = group_ids(X)
GRID = dict(lr_C=[0.1, 1.0, 10.0], rf_depth=[3, 6, 10], xgb_n=[200, 800], meta_C=[0.1, 1.0, 10.0])

def oof_base(Xtr, ytr, seed, xgb_n, rf_depth):
    """Fold-isolated OOF base probabilities + pass-through (Algorithm 1, steps 1-11)."""
    Xtr = Xtr.reset_index(drop=True); cats = vocab(Xtr); n = len(Xtr)
    sk = StratifiedGroupKFold(5, shuffle=True, random_state=seed); P_rf = np.zeros((n,3)); P_x = np.zeros((n,3)); T = None
    for a, b in sk.split(Xtr, ytr, group_ids(Xtr)):
        pre = make_pre(cats).fit(Xtr.iloc[a]); A, B = pre.transform(Xtr.iloc[a]), pre.transform(Xtr.iloc[b])
        if T is None: T = np.zeros((n, B.shape[1]))
        P_rf[b] = RandomForestClassifier(100, max_depth=rf_depth, min_samples_split=5, random_state=seed, n_jobs=1).fit(A, ytr[a]).predict_proba(B)
        P_x[b] = xgb(seed, n_estimators=xgb_n).fit(A, ytr[a]).predict_proba(B); T[b] = B
    return T, P_rf, P_x

def fit_pred(Xtr, ytr, Xte, seed, xgb_n, rf_depth, meta_C, lr_C):
    cats = vocab(Xtr); pre = make_pre(cats).fit(Xtr); A, B = pre.transform(Xtr), pre.transform(Xte)
    rfm = RandomForestClassifier(100, max_depth=rf_depth, min_samples_split=5, random_state=seed, n_jobs=1).fit(A, ytr)
    xm = xgb(seed, n_estimators=xgb_n).fit(A, ytr)
    out = dict(RF=rfm.predict_proba(B), XGB=xm.predict_proba(B)); out["SoftVote(RF+XGB)"] = (out["RF"] + out["XGB"]) / 2
    out["LogReg"] = LogisticRegression(C=lr_C, max_iter=5000).fit(A, ytr).predict_proba(B)
    T, Prf, Px = oof_base(Xtr, ytr, seed, xgb_n, rf_depth)
    meta = LogisticRegression(C=meta_C, max_iter=5000).fit(np.hstack([T, Prf, Px]), ytr)
    out["Stack(RF+XGB)"] = meta.predict_proba(np.hstack([B, out["RF"], out["XGB"]]))
    return out

def inner_select(Xtr, ytr, seed):
    Xtr = Xtr.reset_index(drop=True); gi = group_ids(Xtr); sk = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
    sc = {k: {} for k in ["lr", "rf", "xgb", "sv", "stack"]}
    def add(d, key, v): d.setdefault(key, []).append(v)
    for a, b in sk.split(Xtr, ytr, gi):
        Xa, Xb, ya, yb = Xtr.iloc[a].reset_index(drop=True), Xtr.iloc[b].reset_index(drop=True), ytr[a], ytr[b]
        cats = vocab(Xa); pre = make_pre(cats).fit(Xa); A, B = pre.transform(Xa), pre.transform(Xb)
        ll = lambda P: log_loss(yb, np.clip(P, 1e-12, 1), labels=[0,1,2])
        for C in GRID["lr_C"]: add(sc["lr"], C, ll(LogisticRegression(C=C, max_iter=5000).fit(A, ya).predict_proba(B)))
        prf = {d: RandomForestClassifier(100, max_depth=d, min_samples_split=5, random_state=seed, n_jobs=1).fit(A, ya).predict_proba(B) for d in GRID["rf_depth"]}
        px = {n: xgb(seed, n_estimators=n).fit(A, ya).predict_proba(B) for n in GRID["xgb_n"]}
        for d in GRID["rf_depth"]: add(sc["rf"], d, ll(prf[d]))
        for n in GRID["xgb_n"]: add(sc["xgb"], n, ll(px[n]))
        for d, n in itertools.product(GRID["rf_depth"], GRID["xgb_n"]): add(sc["sv"], (d, n), ll((prf[d] + px[n]) / 2))
        for n in GRID["xgb_n"]:                                           # stack: meta-C and XGB rounds, RF depth fixed at 10 for cost
            T, Prf, Px = oof_base(Xa, ya, seed, n, 10); Z = np.hstack([T, Prf, Px])
            Zb = np.hstack([B, prf[10], px[n]])
            for mc in GRID["meta_C"]: add(sc["stack"], (n, mc), ll(LogisticRegression(C=mc, max_iter=5000).fit(Z, ya).predict_proba(Zb)))
    best = {k: min(v, key=lambda c: np.mean(v[c])) for k, v in sc.items()}
    return best

def task(rep, fold, tr, te):
    seed = 2000 + rep; Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
    b = inner_select(Xtr, y[tr], seed)
    # model-specific tuned configs
    res = {}
    cfgs = {"LogReg": dict(xgb_n=800, rf_depth=10, meta_C=1, lr_C=b["lr"]), "RF": dict(xgb_n=800, rf_depth=b["rf"], meta_C=1, lr_C=1),
            "XGB": dict(xgb_n=b["xgb"], rf_depth=10, meta_C=1, lr_C=1), "SoftVote(RF+XGB)": dict(xgb_n=b["sv"][1], rf_depth=b["sv"][0], meta_C=1, lr_C=1),
            "Stack(RF+XGB)": dict(xgb_n=b["stack"][0], rf_depth=10, meta_C=b["stack"][1], lr_C=1)}
    for m, c in cfgs.items():
        res[m] = fit_pred(Xtr, y[tr], Xte, seed, **c)[m]
    return rep, fold, te, res, b

jobs = []
for rep in range(R):
    for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(X, y, g)): jobs.append((rep, fold, tr, te))
t0 = time.time(); out = Parallel(n_jobs=2)(delayed(task)(*j) for j in jobs); print(proto, "nested compute s", round(time.time()-t0))
rows, preds, sel = [], [], []
for rep, fold, te, P, b in out:
    sel.append(dict(protocol=proto, repeat=rep, fold=fold, **{k: str(v) for k, v in b.items()}))
    for m, p in P.items():
        yt = y[te]; yh = p.argmax(1); Y = np.eye(3)[yt]
        rows.append(dict(protocol=proto, repeat=rep, fold=fold, model=m, acc=accuracy_score(yt, yh), macro_f1=f1_score(yt, yh, average="macro", labels=[0,1,2]),
                         brier=float(((p - Y)**2).sum(1).mean()), logloss=log_loss(yt, np.clip(p,1e-12,1), labels=[0,1,2])))
        for i, idx in enumerate(te): preds.append(dict(protocol=proto, repeat=rep, fold=fold, model=m, row=int(idx), y=int(y[idx]), pred=int(yh[i]), p_insomnia=p[i,0], p_none=p[i,1], p_apnea=p[i,2]))
pd.DataFrame(rows).to_csv(fRESDIR + "nested_folds_{proto}.csv", index=False)
pd.DataFrame(preds).to_csv(fRESDIR + "nested_oof_predictions_{proto}.csv", index=False)
pd.DataFrame(sel).to_csv(fRESDIR + "nested_selected_{proto}.csv", index=False)
d = pd.DataFrame(rows); print(d.groupby("model")[["acc","macro_f1","brier","logloss"]].agg(["mean","std"]).round(3))
