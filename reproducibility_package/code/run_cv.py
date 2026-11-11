"""Repeated stratified CV under identical conditions for all models.
usage: run_cv.py <protocol> <repeats>
 protocols: grouped  (374 rows, duplicates kept together)  <- primary
            dedup    (132 unique rows, plain stratified)
            naive    (374 rows, plain stratified; replicates the leakage-prone original style)
"""
import sys, time, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, balanced_accuracy_score
from common import *

proto, R = sys.argv[1], int(sys.argv[2]); K = 5
X, y = load(); g = group_ids(X)
if proto == "dedup":
    keep = ~X.assign(y=y).duplicated().values
    X, y = X[keep].reset_index(drop=True), y[keep]; g = group_ids(X)
CORE = ["Majority","LogReg","Tree(d=3)","RF","XGB","SoftVote(RF+XGB)","Stack(RF+XGB)"]
EXTRA = ["ExtraTrees","HistGB","LightGBM","CatBoost","MLP","SVM(rbf)"]
MODELS = CORE if proto == "naive" else CORE + EXTRA

def fit_all(Xtr, ytr, Xte, seed):
    out = {}; cats = vocab(Xtr); pre = make_pre(cats).fit(Xtr); A, B = pre.transform(Xtr), pre.transform(Xte)
    def put(name, m): out[name] = m.fit(A, ytr).predict_proba(B)
    out["Majority"] = np.tile(np.bincount(ytr, minlength=3)/len(ytr), (len(Xte),1))
    put("LogReg", LogisticRegression(C=1.0, max_iter=5000))
    put("Tree(d=3)", DecisionTreeClassifier(max_depth=3, random_state=seed))
    r = rf(seed).fit(A, ytr); x = xgb(seed).fit(A, ytr)
    out["RF"], out["XGB"] = r.predict_proba(B), x.predict_proba(B)
    out["SoftVote(RF+XGB)"] = (out["RF"] + out["XGB"]) / 2        # same fitted base learners as RF / XGB rows
    out["Stack(RF+XGB)"] = OOFStack(seed=seed).fit(Xtr, ytr).predict_proba(Xte)
    if proto != "naive":
        put("ExtraTrees", ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=1))
        put("HistGB", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed))
        put("LightGBM", LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=8, min_child_samples=5,
                                       random_state=seed, verbose=-1, n_jobs=1))
        put("CatBoost", CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, verbose=0,
                                           thread_count=1, random_seed=seed))
        put("MLP", MLPClassifier((32,16), alpha=1e-3, max_iter=3000, random_state=seed))
        put("SVM(rbf)", SVC(C=2.0, probability=True, random_state=seed))
    return out

def task(rep, fold, tr, te):
    seed = 1000 + rep
    return rep, fold, te, fit_all(X.iloc[tr].reset_index(drop=True), y[tr], X.iloc[te].reset_index(drop=True), seed)

jobs, assign = [], []
for rep in range(R):
    if proto == "grouped": sp = StratifiedGroupKFold(K, shuffle=True, random_state=rep).split(X, y, g)
    else:                  sp = StratifiedKFold(K, shuffle=True, random_state=rep).split(X, y)
    for fold, (tr, te) in enumerate(sp):
        jobs.append((rep, fold, tr, te)); assign += [(rep, fold, int(i)) for i in te]
t0 = time.time()
res = Parallel(n_jobs=2, verbose=0)(delayed(task)(*j) for j in jobs)
print(proto, "compute s:", round(time.time()-t0))

rows, preds = [], []
for rep, fold, te, P in res:
    for m in MODELS:
        p = P[m]; yt = y[te]; yh = p.argmax(1)
        Y = np.eye(3)[yt]
        rows.append(dict(protocol=proto, repeat=rep, fold=fold, model=m, n_test=len(te),
            acc=accuracy_score(yt, yh), macro_f1=f1_score(yt, yh, average="macro", labels=[0,1,2]),
            bal_acc=balanced_accuracy_score(yt, yh),
            logloss=log_loss(yt, np.clip(p,1e-12,1), labels=[0,1,2]),
            brier=float(((p - Y)**2).sum(1).mean()),
            auc=roc_auc_score(yt, p, multi_class="ovr", average="macro", labels=[0,1,2])))
        for i, idx in enumerate(te):
            preds.append(dict(protocol=proto, repeat=rep, fold=fold, model=m, row=int(idx), y=int(y[idx]),
                              pred=int(yh[i]), p_insomnia=p[i,0], p_none=p[i,1], p_apnea=p[i,2]))
pd.DataFrame(rows).to_csv(fRESDIR + "folds_{proto}.csv", index=False)
pd.DataFrame(preds).to_csv(fRESDIR + "oof_predictions_{proto}.csv", index=False)
pd.DataFrame(assign, columns=["repeat","fold","row"]).to_csv(fRESDIR + "split_ids_{proto}.csv", index=False)
d = pd.DataFrame(rows).groupby("model")[["acc","macro_f1","brier","logloss","auc"]].agg(["mean","std"]).round(3)
print(d.loc[MODELS])
