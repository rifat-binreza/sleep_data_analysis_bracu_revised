"""Explain the near-equal train/test accuracy: with duplicated rows and conflicting labels, in-sample accuracy is capped by the label-noise ceiling."""
import json, numpy as np
from common import *
X, y = load(); g = group_ids(X)
cats = vocab(X); pre = make_pre(cats).fit(X); A = pre.transform(X)
out = {}
for name, m in [("RF", rf(42)), ("XGB", xgb(42))]:
    out[name + "_train_acc_all374"] = float((m.fit(A, y).predict(A) == y).mean())
st = OOFStack(seed=42).fit(X, y); out["Stack_train_acc_all374"] = float((st.predict(X) == y).mean())
import pandas as pd
d = pd.DataFrame({"g": g, "y": y}); out["ceiling_all374"] = float(d.groupby("g")["y"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(d))
print(json.dumps(out, indent=1)); json.dump(out, open(RESDIR + "train_gap.json", "w"), indent=1)
