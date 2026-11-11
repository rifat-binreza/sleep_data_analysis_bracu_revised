import json, numpy as np
from sklearn.model_selection import train_test_split
from common import *
X, y = load(); tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=42)
Xd = X.iloc[tr].reset_index(drop=True); yd = y[tr]; st = OOFStack(seed=42).fit(Xd, yd)
f = lambda D: st.predict_proba(D)[:, 0]
a, b, K = "Sleep Duration", "Quality of Sleep", 6
ea = np.unique(np.quantile(Xd[a].values, np.linspace(0, 1, K + 1))); eb = np.unique(np.quantile(Xd[b].values, np.linspace(0, 1, K + 1))); na, nb = len(ea)-1, len(eb)-1
def ale(Z0):
    ia = np.clip(np.searchsorted(ea, Z0[a].values, side="right")-1, 0, na-1); ib = np.clip(np.searchsorted(eb, Z0[b].values, side="right")-1, 0, nb-1)
    D = np.zeros((na, nb)); N = np.zeros((na, nb))
    for i in range(na):
        for j in range(nb):
            m = (ia == i) & (ib == j)
            if not m.any(): continue
            Z = Z0[m]
            def ev(x, y_): T = Z.copy(); T[a] = x; T[b] = y_; return f(T)
            D[i, j] = np.mean(ev(ea[i+1], eb[j+1]) - ev(ea[i], eb[j+1]) - ev(ea[i+1], eb[j]) + ev(ea[i], eb[j])); N[i, j] = m.sum()
    A = np.pad(np.cumsum(np.cumsum(D, 0), 1), ((1, 0), (1, 0)))
    Wc = np.zeros_like(A)
    for i in range(na+1):
        for j in range(nb+1): Wc[i, j] = N[max(i-1, 0):i+1, max(j-1, 0):j+1].sum()
    return A - (A*Wc).sum()/max(Wc.sum(), 1), N
A0, N = ale(Xd)
rng = np.random.default_rng(0); B = []
for _ in range(200):
    A, _n = ale(Xd.iloc[rng.integers(0, len(Xd), len(Xd))].reset_index(drop=True)); B.append(A)
B = np.array(B); lo, hi = np.percentile(B, 2.5, 0), np.percentile(B, 97.5, 0); sd = B.std(0)
nz = N[N > 0]
out = dict(cells=int(N.size), occupied=int((N > 0).sum()), min_n=int(nz.min()), median_n=float(np.median(nz)), ge10=int((N >= 10).sum()), share_ge10=float((N[N>0] >= 10).mean()),
           share_records_in_ge10=float(N[N >= 10].sum()/N.sum()), band_halfwidth_mean=float(((hi-lo)/2).mean()), band_halfwidth_max=float(((hi-lo)/2).max()), sd_mean=float(sd.mean()),
           ale_range=float(A0.max()-A0.min()), excl0=int(((lo > 0) | (hi < 0)).sum()), ncorner=int(A0.size), N=N.tolist())
json.dump(out, open("../results/ale_unc.json", "w"), indent=1); print({k: v for k, v in out.items() if k != 'N'}); print(N)
