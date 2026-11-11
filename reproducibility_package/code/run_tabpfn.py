"""Evaluate pretrained TabPFN under the identical outer splits of run_cv.py.  usage: python run_tabpfn.py grouped 10
Requires: pip install tabpfn  (weights are downloaded on first use; network access to HuggingFace needed)."""
import sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score
from tabpfn import TabPFNClassifier
from common import *
proto, Rr = sys.argv[1], int(sys.argv[2]); X, y = load(); g = group_ids(X)
if proto == "dedup":
    keep = ~X.assign(y=y).duplicated().values; X, y = X[keep].reset_index(drop=True), y[keep]; g = group_ids(X)
pre = make_pre(); rows = []
for rep in range(Rr):
    sp = StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(X, y, g) if proto == "grouped" else StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y)
    for fold, (tr, te) in enumerate(sp):
        P = make_pre().fit(X.iloc[tr]); m = TabPFNClassifier(random_state=1000 + rep).fit(P.transform(X.iloc[tr]), y[tr]); yh = m.predict(P.transform(X.iloc[te]))
        rows.append(dict(protocol=proto, repeat=rep, fold=fold, acc=accuracy_score(y[te], yh), macro_f1=f1_score(y[te], yh, average="macro")))
d = pd.DataFrame(rows); d.to_csv(f"tabpfn_{proto}.csv", index=False); print(d[["acc", "macro_f1"]].mean())
