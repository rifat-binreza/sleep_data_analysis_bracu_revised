"""Random-split sensitivity of the PRIMARY two-learner stack and soft voting: 13 stratified 80:20 split seeds (50..650)."""
import numpy as np, pandas as pd
from scipy.stats import beta
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from common import *
X, y = load(); g = group_ids(X)
def one(seed):
    tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=seed)
    Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True); pre = make_pre(vocab(Xtr)).fit(Xtr); A, B = pre.transform(Xtr), pre.transform(Xte)
    r = rf(42).fit(A, y[tr]); x = xgb(42).fit(A, y[tr]); st = OOFStack(seed=42).fit(Xtr, y[tr]); out = []
    novel = np.mean([g[i] not in set(g[tr]) for i in te])
    for name, ptr, pte in [("Soft voting (RF+XGB)", (r.predict_proba(A) + x.predict_proba(A)) / 2, (r.predict_proba(B) + x.predict_proba(B)) / 2), ("Stack (RF+XGB)", st.predict_proba(Xtr), st.predict_proba(Xte))]:
        kt, ke = int((ptr.argmax(1) == y[tr]).sum()), int((pte.argmax(1) == y[te]).sum()); nt, ne = len(tr), len(te)
        cp = lambda k, n: (beta.ppf(.025, k, n - k + 1) if k else 0., beta.ppf(.975, k + 1, n - k) if k < n else 1.)
        out.append(dict(seed=seed, model=name, train_acc=kt / nt, test_acc=ke / ne, train_lo=cp(kt, nt)[0], train_hi=cp(kt, nt)[1], test_lo=cp(ke, ne)[0], test_hi=cp(ke, ne)[1], test_correct=ke, test_n=ne, share_test_vectors_seen_in_train=1 - novel))
    return out
rows = sum(Parallel(n_jobs=2)(delayed(one)(s) for s in range(50, 651, 50)), [])
D = pd.DataFrame(rows); D.to_csv("../results/seed_sweep.csv", index=False)
print(D.groupby("model")[["train_acc", "test_acc", "share_test_vectors_seen_in_train"]].agg(["mean", "std"]).round(4).to_string())
