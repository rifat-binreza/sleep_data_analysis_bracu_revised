import numpy as np, pandas as pd
from math import comb
from common import *
R = RESDIR; rng = np.random.default_rng(11); out = []; summ = []
def mcn(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n)
for proto in ["grouped", "dedup"]:
    X, y = load()
    if proto == "dedup": keep = ~X.assign(y=y).duplicated().values; X, y = X[keep].reset_index(drop=True), y[keep]
    g = group_ids(X); ug, inv = np.unique(g, return_inverse=True); ib = [np.where(inv == k)[0] for k in range(len(ug))]
    f = pd.read_csv(R + f"nested_folds_{proto}.csv"); p = pd.read_csv(R + f"nested_oof_predictions_{proto}.csv")
    for m, d in f.groupby("model"):
        rep = d.groupby("repeat")[["acc", "macro_f1", "brier", "logloss"]].mean()
        t = p[p.model == m].copy(); t["c"] = (t.pred == t.y).astype(int); ra = t.pivot(index="row", columns="repeat", values="c").sort_index().mean(1).values
        bs = [ra[np.concatenate([ib[k] for k in rng.integers(0, len(ug), len(ug))])].mean() for _ in range(5000)]
        summ.append(dict(protocol=proto, model=m, **{k: rep[k].mean() for k in rep}, **{k + "_sd": rep[k].std(ddof=1) for k in rep}, acc_bs_lo=np.percentile(bs, 2.5), acc_bs_hi=np.percentile(bs, 97.5)))
    def corr(m):
        t = p[p.model == m].copy(); t["c"] = (t.pred == t.y).astype(int); return t.pivot(index="row", columns="repeat", values="c").sort_index()
    cs = corr("Stack(RF+XGB)")
    for m in ["LogReg", "RF", "XGB", "SoftVote(RF+XGB)"]:
        cm = corr(m); rd = (cs.values - cm.values).mean(1)
        bs = [rd[np.concatenate([ib[k] for k in rng.integers(0, len(ug), len(ug))])].mean() for _ in range(20000)]
        ps = [mcn(int(((cs[r].values == 1) & (cm[r].values == 0)).sum()), int(((cs[r].values == 0) & (cm[r].values == 1)).sum())) for r in cs.columns]
        out.append(dict(protocol=proto, comparator=m, diff_pp=100 * rd.mean(), lo=100 * np.percentile(bs, 2.5), hi=100 * np.percentile(bs, 97.5), mcnemar_p_median=float(np.median(ps)), repeats_p_lt_05=int(np.sum(np.array(ps) < .05))))
pd.DataFrame(summ).to_csv(R + "nested_summary.csv", index=False); pd.DataFrame(out).to_csv(R + "nested_paired.csv", index=False)
pd.set_option("display.width", 200); print(pd.DataFrame(summ).round(3).to_string()); print(pd.DataFrame(out).round(3).to_string())
