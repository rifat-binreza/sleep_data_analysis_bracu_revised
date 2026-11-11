"""Regenerates every model-dependent display of the single 80:20 hold-out (seed 42) from one script."""
import json, itertools, warnings, numpy as np, pandas as pd, shap, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from lime.lime_tabular import LimeTabularExplainer
from common import *
warnings.filterwarnings("ignore")
R = "../results/"; O = "../figs/"; DEC = ["Insomnia", "No recorded disorder", "Sleep apnea"]
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .8, "pdf.fonttype": 42})
BL, OR, GN, RD, GY, TL = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#7f7f7f", "#2a9d8f"
X, y = load(); g = group_ids(X)
tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=42)
Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True); ytr, yte = y[tr], y[te]
pre = make_pre(vocab(Xtr)).fit(Xtr); A, B = pre.transform(Xtr), pre.transform(Xte)
st = OOFStack(seed=42).fit(Xtr, ytr)
f_stack = lambda Z: st.meta.predict_proba(np.hstack([Z, st.rf_.predict_proba(Z), st.xgb_.predict_proba(Z)]))
f_vote = lambda Z: (st.rf_.predict_proba(Z) + st.xgb_.predict_proba(Z)) / 2
Ps, Pv = f_stack(B), f_vote(B); out = {}
# ---- metrics
p, r, f, s = precision_recall_fscore_support(yte, Ps.argmax(1), labels=[0, 1, 2], zero_division=0)
mac = precision_recall_fscore_support(yte, Ps.argmax(1), average="macro"); wav = precision_recall_fscore_support(yte, Ps.argmax(1), average="weighted")
vm = precision_recall_fscore_support(yte, Pv.argmax(1), average="macro")
out["stack"] = dict(per_class=[dict(c=DEC[i], p=p[i], r=r[i], f=f[i], n=int(s[i])) for i in range(3)], macro=list(mac[:3]), weighted=list(wav[:3]), correct=int((Ps.argmax(1) == yte).sum()), cm=confusion_matrix(yte, Ps.argmax(1)).tolist())
out["vote"] = dict(macro=list(vm[:3]), correct=int((Pv.argmax(1) == yte).sum()), cm=confusion_matrix(yte, Pv.argmax(1)).tolist())
# ---- confusion matrices (Fig. confusions)
cm_cmap = LinearSegmentedColormap.from_list("b", ["#f4f8fc", "#9ec5e4", "#08306b"])
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw=dict(wspace=.45))
for ax, P, t, k in zip(axs, [Pv, Ps], ["(a) RF–XGBoost soft voting", "(b) RF–XGBoost stack"], ["vote", "stack"]):
    cm = np.array(out[k]["cm"]); ax.imshow(cm, cmap=cm_cmap, vmin=0, vmax=cm.max()); acc = cm.trace() / cm.sum()
    for i, j in itertools.product(range(3), range(3)): ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11, color="white" if cm[i, j] > cm.max() * .55 else "#111")
    ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(["Insomnia", "No recorded\ndisorder", "Sleep\napnea"], fontsize=8); ax.set_yticklabels(["Insomnia", "No recorded\ndisorder", "Sleep\napnea"], fontsize=8)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Reference class"); ax.set_title(f"{t}\nAccuracy = {100*acc:.2f}% ({cm.trace()}/{cm.sum()})", fontsize=9)
    for sp in ax.spines.values(): sp.set_visible(False)
plt.savefig(O + "confusion_matrices.png", dpi=220, bbox_inches="tight"); plt.close()
# ---- ROC (stack)
fig, ax = plt.subplots(figsize=(5.2, 4.2)); Y = np.eye(3)[yte]; cols = [BL, OR, GN]; aucs = []
for c in range(3):
    fpr, tpr, _ = roc_curve(Y[:, c], Ps[:, c]); a = auc(fpr, tpr); aucs.append(a); ax.plot(fpr, tpr, color=cols[c], lw=2, label=f"{DEC[c]} (AUC = {a:.3f})")
fpr, tpr, _ = roc_curve(Y.ravel(), Ps.ravel()); mi = auc(fpr, tpr); ax.plot(fpr, tpr, "k:", lw=2.4, label=f"Micro-average (AUC = {mi:.3f})")
ax.plot([0, 1], [0, 1], "--", color=GY, lw=1, label="Random classifier (AUC = 0.500)"); ax.set_xlim(-.01, 1); ax.set_ylim(0, 1.03); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.grid(color="#e6e6e6", lw=.7); ax.legend(frameon=False, loc="lower right", fontsize=8)
mac_auc = float(np.mean(aucs)); w_auc = float(np.average(aucs, weights=np.bincount(yte)))
ax.set_title("Stacking classifier: one-vs-rest ROC curves", fontsize=9.5); fig.text(.5, -.01, f"Macro-average ROC–AUC = {mac_auc:.3f}   |   Weighted ROC–AUC = {w_auc:.3f}", ha="center", fontsize=8.5, fontweight="bold")
plt.savefig(O + "stacking_roc_auc.png", dpi=220, bbox_inches="tight"); plt.close(); out["roc"] = dict(per_class=aucs, micro=mi, macro=mac_auc, weighted=w_auc)
for nm, P in [("vote", Pv)]: out["roc_" + nm] = dict(macro=float(roc_auc_score(yte, P, multi_class="ovr")))
# ---- paired comparison
a_ = (Ps.argmax(1) == yte); b_ = (Pv.argmax(1) == yte); n11, n10, n01, n00 = int((b_ & a_).sum()), int((b_ & ~a_).sum()), int((~b_ & a_).sum()), int((~b_ & ~a_).sum())
rng = np.random.default_rng(1); bs = np.array([(a_[i].astype(int) - b_[i].astype(int)).mean() for i in (rng.integers(0, len(yte), len(yte)) for _ in range(20000))])
from math import comb
nn = n10 + n01; mc = 1.0 if nn == 0 else min(1.0, 2 * sum(comb(nn, i) for i in range(0, min(n10, n01) + 1)) / 2 ** nn)
out["paired"] = dict(vote_correct_stack_correct=n11, vote_correct_stack_wrong=n10, vote_wrong_stack_correct=n01, both_wrong=n00, diff_pp=100 * (a_.mean() - b_.mean()), ci=[100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)], mcnemar_p=mc)
# ---- raw-feature explainer helper (readable LIME: raw units, categorical names)
CATV = {c: sorted(Xtr[c].unique()) for c in CAT}; names_raw = FEATS
def encode(df): return np.column_stack([df[c].map({v: i for i, v in enumerate(CATV[c])}).values if c in CAT else df[c].values for c in FEATS]).astype(float)
def decode(M):
    d = pd.DataFrame(M, columns=FEATS)
    for c in CAT: d[c] = d[c].round().astype(int).map(dict(enumerate(CATV[c])))
    for c in FEATS:
        if c not in CAT: d[c] = d[c].astype(float)
    return d
cat_idx = [FEATS.index(c) for c in CAT]; cat_names = {FEATS.index(c): CATV[c] for c in CAT}
wrap = lambda f: (lambda M: f(pre.transform(decode(M))))
# ---- audited record for figures: Person ID 244 (insomnia, correctly classified by both models)
pos = int(np.where(te == 243)[0][0]); lab = 0
lime_fig = {}
for k, f in [("stack", f_stack), ("vote", f_vote)]:
    ex = LimeTabularExplainer(encode(Xtr), feature_names=FEATS, categorical_features=cat_idx, categorical_names=cat_names, class_names=CLASSES, mode="classification", discretize_continuous=True, random_state=0)
    e = ex.explain_instance(encode(Xte)[pos], wrap(f), num_features=10, labels=(lab,), num_samples=5000); w = e.as_list(label=lab)
    lime_fig[k] = dict(w=w, r2=float(e.score[lab] if isinstance(e.score, dict) else e.score), p=f(B[pos:pos + 1])[0].tolist())
fig, axs = plt.subplots(1, 2, figsize=(7.4, 4.0), gridspec_kw=dict(wspace=1.35))
for ax, k, t in zip(axs, ["stack", "vote"], ["(a) Stacking classifier", "(b) RF–XGBoost soft voting"]):
    w = lime_fig[k]["w"][::-1]; ax.barh(range(len(w)), [v for _, v in w], color=[GN if v > 0 else RD for _, v in w], height=.7, edgecolor="white")
    ax.set_yticks(range(len(w))); ax.set_yticklabels([n for n, _ in w], fontsize=7.2); ax.axvline(0, color="#333", lw=.8); ax.grid(axis="x", color="#e6e6e6", lw=.7); ax.set_xlabel("Local surrogate weight (class: Insomnia)", fontsize=8)
    pr = lime_fig[k]["p"]; ax.set_title(f"{t}\nPerson ID 244 (reference: Insomnia)\nP(Insomnia) = {pr[0]:.2f}; surrogate $R^2$ = {lime_fig[k]['r2']:.2f}", fontsize=8.5, loc="left")
plt.savefig(O + "567.png", dpi=220, bbox_inches="tight"); plt.close()
# voting-only copy of panel (b) for the original two-panel figure layout
fig, ax = plt.subplots(figsize=(4.6, 3.4)); w = lime_fig["vote"]["w"][::-1]
ax.barh(range(len(w)), [v for _, v in w], color=[GN if v > 0 else RD for _, v in w], height=.7, edgecolor="white"); ax.set_yticks(range(len(w))); ax.set_yticklabels([n for n, _ in w], fontsize=8); ax.axvline(0, color="#333", lw=.8); ax.grid(axis="x", color="#e6e6e6", lw=.7)
ax.set_xlabel("Local surrogate weight (class: Insomnia)"); ax.set_title(f"Soft voting; Person ID 244 (reference: Insomnia)\nP(Insomnia) = {lime_fig['vote']['p'][0]:.2f}; surrogate $R^2$ = {lime_fig['vote']['r2']:.2f}", fontsize=8.5, loc="left")
plt.savefig(O + "voting_lime_sample0.png", dpi=220, bbox_inches="tight"); plt.close(); out["lime_fig"] = lime_fig
# ---- global SHAP (voting): mean over the 10 independent runs
S = pd.read_csv(R + "shap_stability_runs.csv"); feats = [c for c in S.columns if c not in ("run", "model")]
mv = S[S.model == "SoftVote"][feats].mean().sort_values(); fig, ax = plt.subplots(figsize=(4.8, 3.6))
ax.barh(range(len(mv)), mv.values, color=TL, height=.7, edgecolor="white"); ax.set_yticks(range(len(mv))); ax.set_yticklabels([c.replace("_", " ") for c in mv.index], fontsize=8); ax.grid(axis="x", color="#e6e6e6", lw=.7)
ax.set_xlabel("Mean |SHAP value| (mean of 10 independent runs)"); ax.set_title("Soft voting: global SHAP importance", fontsize=9.5)
plt.savefig(O + "voting_shap_importance.png", dpi=220, bbox_inches="tight"); plt.close(); out["vote_shap_order"] = mv.sort_values(ascending=False).round(4).to_dict()
# ---- stack beeswarm (insomnia) from run 0 sample
names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]; rng = np.random.default_rng(7)
bg = A[rng.choice(len(A), 25, replace=False)]; exs = B[rng.choice(len(B), 35, replace=False)]
phi = shap.PermutationExplainer(f_stack, bg, seed=0)(exs, max_evals=2 * A.shape[1] + 1).values
expl = shap.Explanation(values=phi[:, :, 0], data=exs, feature_names=names)
plt.figure(figsize=(5.2, 5.0)); shap.plots.beeswarm(expl, max_display=15, show=False, plot_size=None); plt.title("Stacking classifier: SHAP beeswarm, class Insomnia", fontsize=9.5)
plt.savefig(O + "stacking_shap_beeswarm_insomnia.png", dpi=220, bbox_inches="tight"); plt.close()
out["beeswarm_top"] = [names[i] for i in np.argsort(-np.abs(phi[:, :, 0]).mean(0))[:8]]
# ---- radar: SHAP (stack, mean of 10 runs) vs LIME (stack, same 35 explanation rows; per-feature sum of |weights|)
orig_of = lambda n: next((c for c in CAT if n.startswith(c + "_")), n)
sh = S[S.model == "Stack"][feats].mean()
idx = rng.choice(len(B), 35, replace=False); lw = np.zeros((35, len(FEATS)))
lex = LimeTabularExplainer(A, feature_names=names, mode="classification", discretize_continuous=False, random_state=0)
for r_, i in enumerate(idx):
    lab_ = int(f_stack(B[i:i + 1]).argmax()); e = lex.explain_instance(B[i], f_stack, num_features=len(names), labels=(lab_,), num_samples=3000); wd = dict(e.as_list(label=lab_))
    for n, v in wd.items(): lw[r_, FEATS.index(orig_of(n.split("__", 1)[1] if "__" in n else n))] += abs(v)
li = pd.Series(lw.mean(0), index=FEATS); order = [f for f in FEATS]
a1, a2 = (sh[order] / sh[order].max()).values, (li[order] / li[order].max()).values; ang = np.linspace(0, 2 * np.pi, len(order), endpoint=False); ang_c = np.r_[ang, ang[0]]
fig = plt.figure(figsize=(5.4, 5.4)); ax = fig.add_subplot(111, polar=True)
for v, c, l in [(a1, "#1f2d4d", "SHAP mean |attribution|"), (a2, OR, "LIME mean |local weight|")]:
    ax.plot(ang_c, np.r_[v, v[0]], "o-", color=c, lw=2, ms=4, label=l); ax.fill(ang_c, np.r_[v, v[0]], color=c, alpha=.15)
ax.set_xticks(ang); ax.set_xticklabels([f.replace("_", " ").replace("Physical Activity Level", "Physical\nActivity") for f in order], fontsize=8); ax.set_rlabel_position(100); ax.set_ylim(0, 1.05); ax.set_yticks([.25, .5, .75, 1]); ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=8); ax.set_title("Stacking classifier: SHAP–LIME global audit", fontsize=9.5, pad=18)
plt.savefig(O + "shap_lime_radar.png", dpi=220, bbox_inches="tight"); plt.close()
out["radar"] = dict(shap=dict(zip(order, a1.round(3))), lime=dict(zip(order, a2.round(3))), spearman=float(pd.Series(a1).corr(pd.Series(a2), method="spearman")))
json.dump(out, open(R + "holdout_figures.json", "w"), indent=1, default=float); print(json.dumps({k: v for k, v in out.items() if k in ("stack", "vote", "roc", "paired", "radar", "beeswarm_top", "vote_shap_order")}, indent=1, default=float)[:3500])
