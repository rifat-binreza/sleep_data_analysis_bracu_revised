"""Predictor-circularity check (R4 minor 1): re-run grouped repeated CV after removing sleep-behaviour predictors."""
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
import common
from common import *
X, y = load(); g = group_ids(X); R_ = 10
SETS = {"All predictors": [], "Without sleep duration & quality": ["Sleep Duration", "Quality of Sleep"],
        "Without all sleep/stress (dur., qual., stress)": ["Sleep Duration", "Quality of Sleep", "Stress Level"]}
def run(setname, drop, rep, fold, tr, te):
    keep = [c for c in FEATS if c not in drop]; Xtr, Xte = X.iloc[tr][keep].reset_index(drop=True), X.iloc[te][keep].reset_index(drop=True)
    num = [c for c in NUM if c in keep]; cat = [c for c in CAT if c in keep]
    mk = lambda: ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, categories=[sorted(Xtr[c].unique()) for c in cat]), cat)])
    pre = mk().fit(Xtr); A, B = pre.transform(Xtr), pre.transform(Xte); seed = 1000 + rep; out = {}
    out["Majority"] = np.full(len(te), np.bincount(y[tr]).argmax())
    out["LogReg"] = LogisticRegression(C=1.0, max_iter=5000).fit(A, y[tr]).predict(B)
    r = rf(seed).fit(A, y[tr]); x = xgb(seed).fit(A, y[tr]); pr, px = r.predict_proba(B), x.predict_proba(B)
    out["RF"], out["XGB"], out["SoftVote(RF+XGB)"] = pr.argmax(1), px.argmax(1), ((pr + px) / 2).argmax(1)
    # stack with this predictor subset (same Algorithm 1, group-aware inner folds)
    n = len(Xtr); gi = group_ids(Xtr); P_rf = np.zeros((n, 3)); P_x = np.zeros((n, 3)); T = None
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xtr, y[tr], gi):
        pr_ = mk().fit(Xtr.iloc[a]); Aa, Bb = pr_.transform(Xtr.iloc[a]), pr_.transform(Xtr.iloc[b]); T = T if T is not None else np.zeros((n, Bb.shape[1]))
        P_rf[b] = rf(seed).fit(Aa, y[tr][a]).predict_proba(Bb); P_x[b] = xgb(seed).fit(Aa, y[tr][a]).predict_proba(Bb); T[b] = Bb
    meta = LogisticRegression(C=2.0, max_iter=5000).fit(np.hstack([T, P_rf, P_x]), y[tr]); out["Stack(RF+XGB)"] = meta.predict(np.hstack([B, pr, px]))
    return setname, rep, fold, te, out
jobs = []
for sn, dr in SETS.items():
    for rep in range(R_):
        for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(X, y, g)): jobs.append((sn, dr, rep, fold, tr, te))
res = Parallel(n_jobs=2)(delayed(run)(*j) for j in jobs); rows = []
for sn, rep, fold, te, out in res:
    for m, p in out.items(): rows.append(dict(set=sn, repeat=rep, fold=fold, model=m, acc=accuracy_score(y[te], p), macro_f1=f1_score(y[te], p, average="macro", labels=[0,1,2])))
d = pd.DataFrame(rows); d.to_csv(RESDIR + "feature_ablation_folds.csv", index=False)
s = d.groupby(["set", "model", "repeat"])[["acc", "macro_f1"]].mean().groupby(["set", "model"]).agg(["mean", "std"]).round(3); s.to_csv(RESDIR + "feature_ablation_summary.csv"); print(s)
