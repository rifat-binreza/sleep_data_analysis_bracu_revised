"""Result figures of the revision, drawn only from released result files.
  fig_models_folds   fold-level accuracy of every model (50 folds) with mean and cluster-bootstrap interval
  fig_diff_heatmap   paired differences, stack minus comparator, overall and by class (run code/perclass_diffs.py first)
  fig_error_anatomy  reference -> prediction flow of the stack and accuracy by label consistency
  fig6_shap          SHAP importance with run-to-run range (legend fixed)
usage: python figs_results.py     (run code/conflict_errors.py first)
"""
import os, json, itertools, numpy as np, pandas as pd
from scipy.stats import spearmanr
from matplotlib.patches import PathPatch, Rectangle, Patch
from matplotlib.path import Path
from matplotlib.lines import Line2D
from style import *

def load(proto):
    p = pd.read_csv(R + f"oof_predictions_{proto}.csv"); s = pd.read_csv(R + "summary_all_protocols.csv"); s = s[s.protocol == proto]
    if os.path.exists(R + f"tabpfn_oof_{proto}.csv"):
        p = pd.concat([p, pd.read_csv(R + f"tabpfn_oof_{proto}.csv")]); t = pd.read_csv(R + "tabpfn_summary.csv"); s = pd.concat([s, t[t.protocol == proto]])
    f = p.assign(c=(p.y == p.pred).astype(float)).groupby(["model", "repeat", "fold"]).c.mean().mul(100).rename("acc").reset_index()
    return f, s.set_index("model")
DATA = {k: load(k) for k in ["grouped", "dedup"]}
models = [m for m in ORDER if m in set(DATA["grouped"][0].model)]
rng = np.random.default_rng(7)

# ---------------- fold-level accuracy of every model
fig, axs = plt.subplots(1, 2, figsize=(7.2, 0.27 * len(models) + 1.0), sharey=True, gridspec_kw=dict(wspace=.07))
for ax, (proto, ttl, lab) in zip(axs, [("grouped", "Grouped CV, 374 records", "(a)"), ("dedup", "Unique rows only, 132 records", "(b)")]):
    f, s = DATA[proto]
    for i, m in enumerate(models):
        yv = len(models) - 1 - i; v = f[f.model == m].acc.values; col = RD if m.startswith("Stack") else (GY if m == "Majority" else BL)
        if m.startswith("Stack"): ax.axhspan(yv - .5, yv + .5, color=RD, alpha=.09, lw=0)
        ax.scatter(v, yv + rng.uniform(-.28, .28, len(v)), s=5, color=col, alpha=.22, lw=0, zorder=2)
        mu, lo, hi = 100 * s.loc[m, "acc"], 100 * s.loc[m, "acc_bs_lo"], 100 * s.loc[m, "acc_bs_hi"]
        ax.plot([lo, hi], [yv, yv], color=INK, lw=1.3, zorder=3, solid_capstyle="butt")
        ax.plot([lo, lo], [yv - .16, yv + .16], color=INK, lw=1.0, zorder=3); ax.plot([hi, hi], [yv - .16, yv + .16], color=INK, lw=1.0, zorder=3)
        ax.scatter([mu], [yv], s=30, marker="D", color=col, edgecolor=INK, lw=.7, zorder=4)
    ax.axvline(100 * s.loc["Majority", "acc"], color=GY, ls=":", lw=1); grid(ax, "x")
    ax.set_xlabel("Accuracy (%)"); ax.set_ylim(-.6, len(models) - .4); ptitle(ax, lab, ttl)
axs[0].axvline(93.3, color=RD, ls="--", lw=.9); axs[0].text(92.7, len(models) - 0.95, "in-sample\nceiling 93.3%", color=RD, fontsize=7, va="center", ha="right")
axs[0].set_yticks(range(len(models))); axs[0].set_yticklabels([NAMES[m] for m in models][::-1]); axs[0].set_xlim(42, 103); axs[1].set_xlim(28, 103)
for t, m in zip(axs[0].get_yticklabels(), models[::-1]):
    if m.startswith("Stack"): t.set_fontweight("bold")
axs[1].tick_params(left=False)
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=BL, alpha=.35, ms=3.5, label="one of 50 folds"),
                    Line2D([], [], marker="D", ls="-", color=INK, mfc=BL, ms=5, lw=1.2, label="mean and 95% cluster-bootstrap interval"),
                    Line2D([], [], ls=":", color=GY, label="majority-class rate")], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.55, -.07), handletextpad=.5, columnspacing=1.6)
save(fig, "fig_models_folds")

# ---------------- paired differences, overall and by class (stack minus comparator)
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
PD_ = pd.read_csv(R + "perclass_diffs.csv"); VMAX = 10.0
cmap = LinearSegmentedColormap.from_list("div", ["#B35806", "#F1A340", "#FBE3C2", "#FFFFFF", "#CFE3F1", "#67A9CF", "#1B6A9E"]); norm = Normalize(-VMAX, VMAX)
GROUPS = [("Simple models", ["Tree(d=3)", "LogReg", "SVM(rbf)"]), ("Neural and pretrained", ["MLP", "FT-Transformer", "TabPFN"]),
          ("Tree ensembles", ["RF", "ExtraTrees", "HistGB", "LightGBM", "CatBoost", "XGB"]), ("Same base learners", ["SoftVote(RF+XGB)"])]
GROUPS = [(t, [m for m in ms if m in set(PD_.comparator)]) for t, ms in GROUPS]; GROUPS = [(t, ms) for t, ms in GROUPS if ms]
METS = [("Accuracy", "Accuracy"), ("Insomnia recall", "Insomnia\nrecall"), ("No-disorder recall", "No recorded\ndisorder recall"), ("Sleep-apnea recall", "Sleep apnea\nrecall")]
RH, SH = 1.0, 0.62; nrow = sum(len(ms) for _, ms in GROUPS); H = nrow * RH + len(GROUPS) * SH
fig, axs = plt.subplots(1, 2, figsize=(7.2, 0.275 * H + 0.95), gridspec_kw=dict(wspace=.035, left=.185, right=.9, top=.9, bottom=.035))
for ax, (proto, ttl, lab) in zip(axs, [("grouped", "Grouped CV, 374 records", "(a)"), ("dedup", "Unique rows only, 132 records", "(b)")]):
    Q = PD_[PD_.protocol == proto].set_index(["comparator", "metric"]); yv = H
    for gt, ms in GROUPS:
        ax.add_patch(Rectangle((0, yv - SH), 4, SH, fc="#d9d9d9", ec="white", lw=1.2)); ax.text(2, yv - SH / 2, gt, ha="center", va="center", fontsize=7.4); yv -= SH
        for m in ms:
            for j, (mk, _) in enumerate(METS):
                r = Q.loc[(m, mk)]; sig = bool(r.excl0); fc = cmap(norm(np.clip(r["diff"], -VMAX, VMAX))) if sig else "white"
                ax.add_patch(Rectangle((j, yv - RH), 1, RH, fc=fc, ec="#bdbdbd", lw=.6))
                dark = sig and abs(r["diff"]) > .62 * VMAX; col = "white" if dark else (INK if sig else "#8a8a8a")
                ax.text(j + .5, yv - .36, f"{r['diff']:+.1f}", ha="center", va="center", fontsize=7.4, fontweight="bold" if sig else "normal", color=col)
                ax.text(j + .5, yv - .74, f"({r.lo:+.1f}, {r.hi:+.1f})", ha="center", va="center", fontsize=5.3, color=col)
            if proto == "grouped": ax.text(-.08, yv - RH / 2, NAMES[m].replace(" (RF+XGB)", ""), ha="right", va="center", fontsize=8)
            yv -= RH
    ax.set_xlim(0, 4); ax.set_ylim(0, H); ax.axis("off")
    for j, (_, ml) in enumerate(METS): ax.text(j + .5, H + .18, ml, ha="center", va="bottom", fontsize=7.2, linespacing=1.05)
    ax.text(0, H + 1.75, lab, fontweight="bold", fontsize=11, ha="left", va="bottom"); ax.annotate(ttl, xy=(0, H + 1.8), xytext=(21, 0), textcoords="offset points", fontsize=9, ha="left", va="bottom", annotation_clip=False)
cax = fig.add_axes([.915, .25, .014, .4]); cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="both"); cb.set_label("Stack minus comparator (pp)", fontsize=7.6); cb.ax.tick_params(labelsize=7); cb.outline.set_linewidth(.5)
save(fig, "fig_diff_heatmap")

# ---------------- error anatomy
C = pd.read_csv(R + "conflict_errors.csv").set_index("model"); F = pd.read_csv(R + "stack_flow_grouped.csv"); M = json.load(open(R + "conflict_meta.json"))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.7), gridspec_kw=dict(width_ratios=[1, 1.0], wspace=1.0))
a = axs[0]; a.axis("off"); gap = 16; xl, xr, bw = 0.0, 1.0, 0.045
def stacks(tot):
    y0 = {}; y = 0
    for c in [2, 1, 0]:
        y0[c] = y; y += tot[c] + gap
    return y0
Lt = F.groupby("y").mean_records.sum(); Rt = F.groupby("pred").mean_records.sum(); L0, R0 = stacks(Lt), stacks(Rt); lo, ro = dict(L0), dict(R0)
for c in [2, 1, 0]:
    a.add_patch(Rectangle((xl - bw, L0[c]), bw, Lt[c], fc=CLASS_COL[c], ec="none")); a.add_patch(Rectangle((xr, R0[c]), bw, Rt[c], fc=CLASS_COL[c], ec="none"))
    a.text(xl - bw - .03, L0[c] + Lt[c] / 2, f"{CLASS_NAME[c]}\n(n = {Lt[c]:.0f})", ha="right", va="center", fontsize=8)
    a.text(xr + bw + .03, R0[c] + Rt[c] / 2, f"{CLASS_NAME[c]}\n({Rt[c]:.1f})", ha="left", va="center", fontsize=8)
for yc in [2, 1, 0]:
    for pc in [2, 1, 0]:
        for cf in [False, True]:
            r = F[(F.y == yc) & (F.pred == pc) & (F.conf == cf)]
            if r.empty: continue
            h = float(r.mean_records.iloc[0]); y1, y2 = lo[yc], ro[pc]; lo[yc] += h; ro[pc] += h
            verts = [(xl, y1), (.5, y1), (.5, y2), (xr, y2), (xr, y2 + h), (.5, y2 + h), (.5, y1 + h), (xl, y1 + h), (xl, y1)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
            wrong = yc != pc
            a.add_patch(PathPatch(Path(verts, codes), fc=CLASS_COL[yc], alpha=.78 if wrong else .30, ec="none"))
            if cf: a.add_patch(PathPatch(Path(verts, codes), fc="none", ec=INK, lw=0, hatch="//////", alpha=.45))
a.set_xlim(-.42, 1.42); a.set_ylim(-6, sum(Lt) + 2 * gap + 6)
a.text(xl - bw / 2, sum(Lt) + 2 * gap + 8, "Reference", ha="center", va="bottom", fontsize=8.5, fontweight="bold"); a.text(xr + bw / 2, sum(Lt) + 2 * gap + 8, "Stack prediction", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
a.legend(handles=[Patch(fc="#bbbbbb", ec="none", alpha=.45, label="correct"), Patch(fc="#555555", ec="none", alpha=.85, label="misclassified"),
                  Patch(fc="white", ec=INK, lw=.4, hatch="//////", label="conflicting-label vector")], frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.01), fontsize=7.4, columnspacing=1.0, handlelength=1.3, handletextpad=.4)
a.text(-.30, 1.07, "(a)", transform=a.transAxes, fontweight="bold", fontsize=11, ha="left", va="bottom", clip_on=False)
b = axs[1]; ms = [m for m in models if m in C.index]
for i, m in enumerate(ms):
    yv = len(ms) - 1 - i
    if m.startswith("Stack"): b.axhspan(yv - .5, yv + .5, color=RD, alpha=.09, lw=0)
    for k, col in [("acc_conflicting", OR), ("acc_consistent", BL)]:
        b.plot([100 * C.loc[m, k + "_lo"], 100 * C.loc[m, k + "_hi"]], [yv, yv], color=col, lw=1.2, alpha=.9); b.scatter([100 * C.loc[m, k]], [yv], s=24, color=col, edgecolor=INK, lw=.6, zorder=3)
b.axvline(100 * M["ceiling_on_conflicting"], color=OR, ls="--", lw=.9)
b.text(100 * M["ceiling_on_conflicting"] - .8, len(ms) - .3, f"ceiling {100*M['ceiling_on_conflicting']:.1f}%", color="#a66f00", fontsize=7.2, ha="right", va="bottom")
b.set_yticks(range(len(ms))); b.set_yticklabels([NAMES[m] for m in ms][::-1]); b.set_xlabel("Grouped-CV accuracy (%)"); b.set_xlim(20, 101); b.set_ylim(-.6, len(ms) + .5); grid(b, "x")
for t, m in zip(b.get_yticklabels(), ms[::-1]):
    if m.startswith("Stack"): t.set_fontweight("bold")
b.legend(handles=[Line2D([], [], marker="o", color=BL, mec=INK, mew=.6, ms=5, lw=1.2, label=f"consistent-label vectors ({M['n_consistent_rows']} records)"),
                  Line2D([], [], marker="o", color=OR, mec=INK, mew=.6, ms=5, lw=1.2, label=f"conflicting-label vectors ({M['n_conflicting_rows']} records)")],
         frameon=False, loc="upper center", bbox_to_anchor=(.42, -.13), fontsize=7.4, ncol=1, handlelength=1.6)
b.text(-.70, 1.07, "(b)", transform=b.transAxes, fontweight="bold", fontsize=11, ha="left", va="bottom")
save(fig, "fig_error_anatomy")

# ---------------- SHAP importance with run-to-run range
Sh = pd.read_csv(R + "shap_stability_runs.csv"); feats = [c for c in Sh.columns if c not in ("run", "model")]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(wspace=.62))
for ax, (m, ttl, lab) in zip(axs, [("Stack", "Stacking", "(a)"), ("SoftVote", "Soft voting", "(b)")]):
    d = Sh[Sh.model == m][feats]; rho = np.mean([spearmanr(d.iloc[i], d.iloc[j])[0] for i, j in itertools.combinations(range(len(d)), 2)])
    mu = d.mean().sort_values(); yy = np.arange(len(mu)); cols = [RD if "BP" in f else BL for f in mu.index]
    ax.barh(yy, mu.values, color=cols, height=.68, edgecolor="white"); ax.hlines(yy, d[mu.index].min().values, d[mu.index].max().values, color=INK, lw=1.0)
    ax.set_yticks(yy); ax.set_yticklabels([f.replace("_BP", " BP") for f in mu.index]); ax.set_xlabel("Mean |SHAP|"); grid(ax, "x")
    ptitle(ax, lab, f"{ttl} (mean rank correlation {rho:.2f})")
fig.legend(handles=[Patch(fc=RD, label="blood-pressure predictors"), Patch(fc=BL, label="other predictors"), Line2D([], [], color=INK, lw=1, label="range over 10 runs")],
           frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5, -.08))
save(fig, "fig6_shap")
print("done")
