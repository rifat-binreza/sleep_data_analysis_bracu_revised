"""SHAP stability (independent backgrounds/explanation samples), block-permutation importance (collinearity), and audited LIME."""
import json, itertools, warnings, numpy as np, pandas as pd, shap
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from lime.lime_tabular import LimeTabularExplainer
from common import *
warnings.filterwarnings("ignore")
R = RESDIR
X, y = load(); g = group_ids(X)
tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=42)     # same split as the original single hold-out
Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True); ytr, yte = y[tr], y[te]
pre = make_pre(vocab(Xtr)).fit(Xtr); A, B = pre.transform(Xtr), pre.transform(Xte)
st = OOFStack(seed=42).fit(Xtr, ytr)
names = list(pre.get_feature_names_out()); names = [n.split("__", 1)[1] for n in names]
def orig_of(n):
    for c in CAT:
        if n.startswith(c + "_"): return c
    return n
orig = [orig_of(n) for n in names]; ofeats = FEATS
def f_stack(Z): return st.meta.predict_proba(np.hstack([Z, st.rf_.predict_proba(Z), st.xgb_.predict_proba(Z)]))
def f_vote(Z): return (st.rf_.predict_proba(Z) + st.xgb_.predict_proba(Z)) / 2
FUN = {"Stack": f_stack, "SoftVote": f_vote}
def importance(phi):                       # phi: (n, p, C); sum one-hot columns of the same original feature, then mean |.| over rows and classes
    out = {}
    for f in ofeats:
        cols = [i for i, o in enumerate(orig) if o == f]; out[f] = float(np.abs(phi[:, cols, :].sum(1)).mean())
    return out
# ---------- SHAP stability: 10 runs, each with a fresh 25-row background (from train) and fresh 35-row explanation sample (from test)
rng = np.random.default_rng(7); RUNS = 10; rows = []
for r in range(RUNS):
    bg = A[rng.choice(len(A), 25, replace=False)]; ex = B[rng.choice(len(B), 35, replace=False)]
    for mname, f in FUN.items():
        ex_ = shap.PermutationExplainer(f, bg, seed=r); phi = ex_(ex, max_evals=2 * A.shape[1] + 1).values
        rows.append(dict(run=r, model=mname, **importance(phi)))
S = pd.DataFrame(rows); S.to_csv(R + "shap_stability_runs.csv", index=False)
res = {}
for m in FUN:
    M = S[S.model == m][ofeats].values; ranks = [spearmanr(M[i], M[j])[0] for i, j in itertools.combinations(range(RUNS), 2)]
    top3 = [set(np.array(ofeats)[np.argsort(-M[i])[:3]]) for i in range(RUNS)]
    jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(top3, 2)]
    res[m] = dict(mean_pairwise_spearman=float(np.mean(ranks)), min_pairwise_spearman=float(np.min(ranks)), mean_top3_jaccard=float(np.mean(jac)),
                  mean_importance={f: float(M[:, k].mean()) for k, f in enumerate(ofeats)}, cv_importance={f: float(M[:, k].std() / M[:, k].mean()) for k, f in enumerate(ofeats)},
                  rank_range={f: [int(min((-M).argsort(1).argsort(1)[:, k]) + 1), int(max((-M).argsort(1).argsort(1)[:, k]) + 1)] for k, f in enumerate(ofeats)})
ms, mv = [np.array([res[m]["mean_importance"][f] for f in ofeats]) for m in ("Stack", "SoftVote")]
res["stack_vs_vote_spearman_of_mean_importance"] = float(spearmanr(ms, mv)[0])
BLOCKS = {"Sleep/stress block": ["Sleep Duration", "Quality of Sleep", "Stress Level"], "Blood-pressure block": ["Systolic_BP", "Diastolic_BP"], "Activity block": ["Physical Activity Level", "Daily Steps"]}
for m in FUN:
    Mdf = S[S.model == m]
    res[m]["block_importance_mean"] = {b: float(Mdf[fs].sum(1).mean()) for b, fs in BLOCKS.items()}
    res[m]["block_importance_cv"] = {b: float(Mdf[fs].sum(1).std() / Mdf[fs].sum(1).mean()) for b, fs in BLOCKS.items()}
# ---------- collinearity: Spearman matrix and VIF on all 374 observations
num = X[NUM]; rho = num.corr(method="spearman"); res["spearman_pairs_abs_ge_0.75"] = {f"{a} | {b}": float(rho.loc[a, b]) for a, b in itertools.combinations(NUM, 2) if abs(rho.loc[a, b]) >= 0.75}
Z = (num - num.mean()) / num.std(); vif = {}
for c in NUM:
    o = [k for k in NUM if k != c]; coef = np.linalg.lstsq(np.c_[np.ones(len(Z)), Z[o].values], Z[c].values, rcond=None)[0]
    resid = Z[c].values - np.c_[np.ones(len(Z)), Z[o].values] @ coef; vif[c] = float(1 / (resid.var() / Z[c].values.var()))
res["vif"] = vif; rho.round(3).to_csv(R + "spearman_all374.csv")
# ---------- single-feature vs block permutation (held-out, grouped 5-fold, 10 permutations): shows how correlated predictors split importance
sk = StratifiedGroupKFold(5, shuffle=True, random_state=0); rp = np.random.default_rng(3); perm_rows = []
for k, (a, b) in enumerate(sk.split(X, y, g)):
    Xa, Xb = X.iloc[a].reset_index(drop=True), X.iloc[b].reset_index(drop=True)
    m = OOFStack(seed=k).fit(Xa, y[a]); P0 = {"Stack": m.predict_proba(Xb), "SoftVote": (m.rf_.predict_proba(m.pre.transform(Xb)) + m.xgb_.predict_proba(m.pre.transform(Xb))) / 2}
    ll = lambda P: -np.log(np.clip(P[np.arange(len(b)), y[b]], 1e-12, 1)).mean()
    units = {f: [f] for f in ofeats}; units.update({"[BLOCK] " + k_: v for k_, v in BLOCKS.items()})
    for u, cols in units.items():
        for _ in range(10):
            idx = rp.permutation(len(Xb)); Xp = Xb.copy(); Xp[cols] = Xb[cols].values[idx]            # permute the block jointly (rows shuffled together)
            Ap = m.pre.transform(Xp); Pp = {"Stack": m.predict_proba(Xp), "SoftVote": (m.rf_.predict_proba(Ap) + m.xgb_.predict_proba(Ap)) / 2}
            for mn in Pp: perm_rows.append(dict(fold=k, model=mn, unit=u, dlogloss=ll(Pp[mn]) - ll(P0[mn])))
PP = pd.DataFrame(perm_rows).groupby(["model", "unit"])["dlogloss"].agg(["mean", "std"]).reset_index(); PP.to_csv(R + "perm_importance_block.csv", index=False)
# ---------- audited LIME on four held-out observations (original split): index, label, probabilities, surrogate fidelity, repeat stability
cls_pred = f_stack(B).argmax(1); pick = []
for c in (0, 1, 2):
    ok = [i for i in range(len(te)) if yte[i] == c and cls_pred[i] == yte[i]]; pick.append(ok[0])
bad = [i for i in range(len(te)) if cls_pred[i] != yte[i]]; pick.append(bad[0] if bad else None); pick = [p for p in pick if p is not None]
lime_rows, lime_stab = [], {}
for i in pick:
    for mname, f in FUN.items():
        w_runs, r2s = [], []
        for s in range(10):
            ex = LimeTabularExplainer(A, feature_names=names, class_names=CLASSES, mode="classification", discretize_continuous=False, random_state=s)
            lab = int(f(B[i:i+1]).argmax())
            e = ex.explain_instance(B[i], f, num_features=len(names), labels=(lab,), num_samples=5000)
            wv = dict(e.as_list(label=lab)); w_runs.append([wv[n] for n in names]); r2s.append(e.score[lab] if isinstance(e.score, dict) else e.score)
        W = np.array(w_runs); top = np.argsort(-np.abs(W.mean(0)))[:5]
        rs = [spearmanr(np.abs(W[a]), np.abs(W[b]))[0] for a, b in itertools.combinations(range(10), 2)]
        p = f(B[i:i+1])[0]
        lime_rows.append(dict(model=mname, test_position=i, csv_row=int(te[i]) + 2, person_id=int(te[i]) + 1, true_label=CLASSES[yte[i]], predicted=CLASSES[int(p.argmax())],
                              p_insomnia=p[0], p_none=p[1], p_apnea=p[2], lime_R2_mean=float(np.mean(r2s)), lime_R2_min=float(np.min(r2s)), weight_rank_spearman_over_seeds=float(np.mean(rs)),
                              top5=";".join(f"{names[t]}({W.mean(0)[t]:+.3f})" for t in top)))
L = pd.DataFrame(lime_rows); L.to_csv(R + "lime_audit.csv", index=False)
json.dump(res, open(R + "explain_results.json", "w"), indent=1)
print(json.dumps({m: {k: v for k, v in res[m].items() if k in ("mean_pairwise_spearman", "min_pairwise_spearman", "mean_top3_jaccard", "block_importance_mean", "block_importance_cv")} for m in FUN}, indent=1))
print("stack-vs-vote spearman", res["stack_vs_vote_spearman_of_mean_importance"]); print(res["vif"]); print(res["spearman_pairs_abs_ge_0.75"])
print(PP.sort_values(["model", "mean"], ascending=[True, False]).round(4).to_string()); print(L.round(3).to_string())
