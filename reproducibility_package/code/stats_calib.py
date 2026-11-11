"""Paired comparisons, uncertainty and calibration from observation-level OOF predictions."""
import json, numpy as np, pandas as pd
from scipy import stats
from math import comb
from common import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = RESDIR
rng = np.random.default_rng(0)

def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); p = sum(comb(n, i) for i in range(0, k+1)) / 2**n * 2
    return min(1.0, p)

def load_proto(proto):
    X, y = load()
    if proto == "dedup":
        keep = ~X.assign(y=y).duplicated().values; X, y = X[keep].reset_index(drop=True), y[keep]
    return X, y, group_ids(X)

def ece(P, y, bins=5):
    conf = P.max(1); hit = (P.argmax(1) == y).astype(float)
    edges = np.quantile(conf, np.linspace(0, 1, bins+1)); edges[-1] += 1e-9; e = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum(): e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return e

summary, paired_rows = [], []
for proto in ["grouped", "dedup", "naive"]:
    X, y, g = load_proto(proto) if proto != "naive" else (*load(), None)
    if proto == "naive": g = group_ids(X)
    fo = pd.read_csv(R + f"folds_{proto}.csv"); pr = pd.read_csv(R + f"oof_predictions_{proto}.csv")
    models = fo["model"].unique().tolist()
    # ---- per-repeat pooled-OOF summary (mean, SD, 95% t-CI across repeats) + fold-level SD
    for m in models:
        f = fo[fo.model == m]
        rep = f.groupby("repeat")[["acc","macro_f1","bal_acc","brier","logloss","auc"]].mean()
        d = dict(protocol=proto, model=m)
        for k in rep.columns:
            mu, sd = rep[k].mean(), rep[k].std(ddof=1); h = stats.t.ppf(.975, len(rep)-1) * sd/np.sqrt(len(rep))
            d[k] = mu; d[k+"_sd_rep"] = sd; d[k+"_lo"] = mu-h; d[k+"_hi"] = mu+h
        d["acc_sd_fold"] = f["acc"].std(ddof=1)
        pm = pr[pr.model == m].copy(); pm["c"] = (pm.pred == pm.y).astype(int); rowacc = pm.pivot(index="row", columns="repeat", values="c").sort_index().mean(1).values
        ug_, inv_ = np.unique(g, return_inverse=True); ib = [np.where(inv_ == k)[0] for k in range(len(ug_))]; bsa = np.empty(5000)
        for b_ in range(5000):
            pk = rng.integers(0, len(ug_), len(ug_)); bsa[b_] = rowacc[np.concatenate([ib[k] for k in pk])].mean()
        d["acc_bs_lo"], d["acc_bs_hi"] = float(np.percentile(bsa, 2.5)), float(np.percentile(bsa, 97.5))
        # ECE from pooled predictions of this model
        p = pr[pr.model == m]; eces = []
        for r_, pp in p.groupby("repeat"):
            pp = pp.sort_values("row"); eces.append(ece(pp[["p_insomnia","p_none","p_apnea"]].values, pp["y"].values))
        d["ece5"] = float(np.mean(eces)); summary.append(d)
    # ---- paired comparisons vs the stack
    pivot = {m: pr[pr.model == m].pivot(index="row", columns="repeat", values="pred") for m in ["Stack(RF+XGB)"]}
    ycol = pr[pr.model == "Stack(RF+XGB)"].drop_duplicates("row").set_index("row")["y"].sort_index()
    def correct(m):
        t = pr[pr.model == m].copy(); t["c"] = (t.pred == t.y).astype(int)
        return t.pivot(index="row", columns="repeat", values="c").sort_index()
    cs = correct("Stack(RF+XGB)")
    ug, inv = np.unique(g, return_inverse=True)
    for m in models:
        if m == "Stack(RF+XGB)": continue
        cm = correct(m); dif = cs.values - cm.values            # rows x repeats
        row_d = dif.mean(1)                                      # repeat-averaged per-row difference
        # cluster bootstrap over predictor-vector groups
        idx_by_g = [np.where(inv == k)[0] for k in range(len(ug))]
        bs = np.empty(20000)
        for b in range(20000):
            pick = rng.integers(0, len(ug), len(ug)); rows = np.concatenate([idx_by_g[k] for k in pick]); bs[b] = row_d[rows].mean()
        # Nadeau-Bengio corrected resampled t over fold-level differences
        fs = fo[fo.model == "Stack(RF+XGB)"].set_index(["repeat","fold"])["acc"]; fm = fo[fo.model == m].set_index(["repeat","fold"])["acc"]
        fd = (fs - fm).values; J = len(fd); ntr_te = 4.0
        s2 = fd.var(ddof=1); tnb = fd.mean() / np.sqrt((1/J + ntr_te) * s2) if s2 > 0 else 0.0
        pnb = 2 * (1 - stats.t.cdf(abs(tnb), J-1))
        # per-repeat exact McNemar
        ps, bs_, cs_ = [], [], []
        for r_ in cs.columns:
            a, bb = cs[r_].values, cm[r_].values
            b01 = int(((a == 1) & (bb == 0)).sum()); c10 = int(((a == 0) & (bb == 1)).sum())
            ps.append(mcnemar_exact(b01, c10)); bs_.append(b01); cs_.append(c10)
        paired_rows.append(dict(protocol=proto, comparator=m, n=len(cs), mean_acc_diff_pp=100*row_d.mean(),
            boot_lo_pp=100*np.percentile(bs, 2.5), boot_hi_pp=100*np.percentile(bs, 97.5),
            nb_t=tnb, nb_p=pnb, folds_stack_better=int((fd > 0).sum()), folds_tie=int((fd == 0).sum()), folds_comp_better=int((fd < 0).sum()),
            mcnemar_p_median=float(np.median(ps)), mcnemar_p_min=float(np.min(ps)), mcnemar_p_max=float(np.max(ps)),
            repeats_p_lt_05=int(np.sum(np.array(ps) < .05)), discordant_stack_only_mean=float(np.mean(bs_)), discordant_comp_only_mean=float(np.mean(cs_))))
S = pd.DataFrame(summary); P = pd.DataFrame(paired_rows)
S.to_csv(R + "summary_all_protocols.csv", index=False); P.to_csv(R + "paired_vs_stack.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(S[["protocol","model","acc","acc_lo","acc_hi","macro_f1","brier","logloss","auc","ece5"]].round(3).to_string())
print(P[["protocol","comparator","mean_acc_diff_pp","boot_lo_pp","boot_hi_pp","nb_p","mcnemar_p_median","repeats_p_lt_05","discordant_stack_only_mean","discordant_comp_only_mean"]].round(3).to_string())

# ---- reliability diagram (grouped protocol) 
fig, axs = plt.subplots(1, 2, figsize=(9.5, 4), gridspec_kw=dict(width_ratios=[1.15, 1]))
pr = pd.read_csv(R + "oof_predictions_grouped.csv")
cols = {"Stack(RF+XGB)":"#1b6ca8","SoftVote(RF+XGB)":"#d17a22","XGB":"#2e8b57","LogReg":"#8e44ad","RF":"#999999"}
for m, c in cols.items():
    p = pr[pr.model == m]; P_ = p[["p_insomnia","p_none","p_apnea"]].values; conf = P_.max(1); hit = (P_.argmax(1) == p["y"].values)
    edges = np.array([0, .5, .7, .85, .95, 1.0001]); xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (conf >= lo) & (conf < hi)
        if mk.sum() >= 20: xs.append(conf[mk].mean()); ys.append(hit[mk].mean())
    axs[0].plot(xs, ys, "o-", color=c, label=m, ms=4, lw=1.4)
axs[0].plot([0.3, 1], [0.3, 1], "k--", lw=.8); axs[0].set_xlabel("Mean predicted top-class probability"); axs[0].set_ylabel("Observed accuracy")
axs[0].set_title("(a) Reliability, grouped CV (10 repeats pooled)", fontsize=9); axs[0].legend(fontsize=7, frameon=False); axs[0].set_xlim(.3, 1.02); axs[0].set_ylim(.3, 1.02)
sg = S[S.protocol == "grouped"].set_index("model").loc[["LogReg","RF","XGB","SoftVote(RF+XGB)","Stack(RF+XGB)","CatBoost","SVM(rbf)"]]
axs[1].barh(range(len(sg)), sg["brier"], xerr=[sg["brier"]-sg["brier_lo"], sg["brier_hi"]-sg["brier"]], color="#7aa6c9", ecolor="k", capsize=2)
axs[1].set_yticks(range(len(sg))); axs[1].set_yticklabels(sg.index, fontsize=8); axs[1].invert_yaxis(); axs[1].set_xlabel("Multiclass Brier score (lower is better)")
axs[1].set_title("(b) Brier score, mean and 95% CI", fontsize=9)
for a in axs: a.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig(RESDIR + "fig_calibration.pdf"); plt.savefig(RESDIR + "fig_calibration.png", dpi=200)
