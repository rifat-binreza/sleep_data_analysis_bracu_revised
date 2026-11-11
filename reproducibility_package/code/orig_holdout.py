"""Re-implementation of the ORIGINAL single hold-out protocol (80:20 stratified, seed 42), with observation-level output,
train/test accuracy, exact CIs, and leakage diagnostics."""
import json, numpy as np, pandas as pd
from scipy.stats import beta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from math import comb
from common import *
R = RESDIR
X, y = load(); g = group_ids(X)
tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=42)
Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
cats = vocab(Xtr); pre = make_pre(cats).fit(Xtr); A, B = pre.transform(Xtr), pre.transform(Xte)
rfm = rf(42).fit(A, y[tr]); xm = xgb(42).fit(A, y[tr])
st = OOFStack(seed=42).fit(Xtr, y[tr])
P = {"RF": (rfm.predict_proba(A), rfm.predict_proba(B)), "XGB": (xm.predict_proba(A), xm.predict_proba(B))}
P["SoftVote(RF+XGB)"] = tuple((P["RF"][i] + P["XGB"][i]) / 2 for i in (0, 1))
P["Stack(RF+XGB)"] = (st.predict_proba(Xtr), st.predict_proba(Xte))
lr = LogisticRegression(C=1.0, max_iter=5000).fit(A, y[tr]); P["LogReg"] = (lr.predict_proba(A), lr.predict_proba(B))
novel = np.array([g[i] not in set(g[tr]) for i in te])            # test rows whose predictor vector never occurs in training
res, obs = {}, []
for m, (ptr, pte) in P.items():
    yh = pte.argmax(1); k = int((yh == y[te]).sum()); lo = beta.ppf(.025, k, len(te)-k+1); hi = beta.ppf(.975, k+1, len(te)-k)
    res[m] = dict(test_acc=k/len(te), correct=k, n=len(te), clopper_pearson_95=[lo, hi], macro_f1=f1_score(y[te], yh, average="macro"),
                  train_acc=float(accuracy_score(y[tr], ptr.argmax(1))), acc_on_novel_test_rows=(float((yh[novel] == y[te][novel]).mean()) if novel.sum() else None),
                  n_novel=int(novel.sum()), confusion=confusion_matrix(y[te], yh, labels=[0,1,2]).tolist())
    for i, idx in enumerate(te): obs.append(dict(model=m, row=int(idx), y=CLASSES[y[idx]], pred=CLASSES[yh[i]], p_insomnia=pte[i,0], p_none=pte[i,1], p_apnea=pte[i,2], novel_vector=bool(novel[i])))
pd.DataFrame(obs).to_csv(R + "original_split_observation_level.csv", index=False)
st2 = OOFStack(seed=42, grouped=False).fit(Xtr, y[tr]); res["stack_plain_stratified_inner_folds"] = dict(correct=int((st2.predict(Xte) == y[te]).sum()), n=len(te))
# paired analysis stack vs soft vote on this split
a = (P["Stack(RF+XGB)"][1].argmax(1) == y[te]); b = (P["SoftVote(RF+XGB)"][1].argmax(1) == y[te])
b01, c10 = int((a & ~b).sum()), int((~a & b).sum()); n = b01 + c10
mc = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b01, c10)+1)) / 2**n)
rng = np.random.default_rng(1); bs = np.array([ (a[i].astype(int) - b[i].astype(int)).mean() for i in (rng.integers(0, len(te), len(te)) for _ in range(20000))])
res["paired_stack_vs_softvote"] = dict(stack_only_correct=b01, vote_only_correct=c10, mcnemar_exact_p=mc, boot_diff_pp=100*(a.mean()-b.mean()), boot_ci_pp=[100*np.percentile(bs, 2.5), 100*np.percentile(bs, 97.5)])
# power: smallest true accuracy difference detectable (80% power, alpha .05) given discordance rate p_d, n=75
from scipy.stats import norm
pdisc = n / len(te) if n else 0.04
res["power_note"] = dict(n=len(te), discordant_rate_observed=pdisc, mde_pp_80power=100 * (norm.ppf(.975) + norm.ppf(.8)) * np.sqrt(pdisc / len(te)))
res["leakage"] = dict(test_rows=len(te), test_rows_with_identical_train_vector=int((~novel).sum()))
json.dump(res, open(R + "original_split_results.json", "w"), indent=1, default=float)
for m in P: print(m, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[m].items() if k != "confusion"})
print(res["paired_stack_vs_softvote"], res["power_note"], res["leakage"])
