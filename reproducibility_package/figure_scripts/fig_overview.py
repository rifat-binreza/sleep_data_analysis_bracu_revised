"""Fig. 1 - study overview, drawn programmatically from the released result files (no image-generation tools).
 (a) benchmark audit: one cell per record   (b) fold-isolated RF-XGBoost stack
 (c) evaluation protocol                     (d) accuracy under each evaluation treatment
usage: python fig_overview.py     (run code/conflict_errors.py first: it writes results/record_flags.csv)
"""
import os, json, numpy as np, pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
from style import *

A = json.load(open(R + "audit.json")); OS = json.load(open(R + "original_split_results.json")); S = pd.read_csv(R + "summary_all_protocols.csv").set_index(["protocol", "model"])
FL = pd.read_csv(R + "record_flags.csv"); NCOMP = 13 + os.path.exists(R + "tabpfn_oof_grouped.csv")
CL = [BL, OR, GN]; CLN = ["insomnia", "no recorded disorder", "sleep apnea"]
MUTE, LINE, SOFT = "#5a5a5a", "#6f6f6f", "#f3f3f1"
PB, PO, PG, PR = "#dbe9f6", "#fdecc8", "#d8f0e6", "#f9ddd0"
FW, FH = 7.2, 6.83
fig = plt.figure(figsize=(FW, FH))

def canvas(x, y, w, h):
    """Axes placed in inches whose data units are 1/100 inch, so shapes keep their proportions."""
    ax = fig.add_axes([x / FW, y / FH, w / FW, h / FH]); ax.set_xlim(0, w * 100); ax.set_ylim(0, h * 100); ax.axis("off"); return ax
def title(ax, lab, ttl, y):
    ax.text(0, y, lab, fontweight="bold", fontsize=11, ha="left", va="bottom"); ax.text(23, y + .8, ttl, fontsize=9.2, ha="left", va="bottom")
def rbox(ax, x, y, w, h, fc="white", ec=LINE, lw=.9, r=4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
def arrow(ax, p, q, ls="-", lw=1.1, col=LINE):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=lw, color=col, ls=ls, shrinkA=0, shrinkB=0))
def grid_cells(ax, x, y, cols, rows, cw, ch, fc, gap=1.0):
    """fc: function (row, col) -> colour.  Returns the bounding box."""
    for r_ in range(rows):
        for c_ in range(cols):
            ax.add_patch(Rectangle((x + c_ * cw, y + (rows - 1 - r_) * ch), cw - gap, ch - gap, fc=fc(r_, c_), ec="none"))
    return x, y, cols * cw - gap, rows * ch - gap
def tree(ax, x, y, w, h, col=GN):
    xs = [x + w / 2, x + w / 4, x + 3 * w / 4, x + w / 8, x + 3 * w / 8, x + 5 * w / 8, x + 7 * w / 8]; ys = [y + h, y + h / 2, y + h / 2, y, y, y, y]
    for a_, b_ in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]: ax.plot([xs[a_], xs[b_]], [ys[a_], ys[b_]], color=col, lw=.8, solid_capstyle="round")
    ax.scatter(xs, ys, s=3.2, color=col, zorder=3, lw=0)
def strips(ax, x, y, w, h, pitch, ec="white"):
    for r_ in range(5):
        for k in range(5): ax.add_patch(Rectangle((x + k * w / 5, y - r_ * pitch), w / 5 - 1, h, fc=OR if k == r_ else "#c9dcee", ec="none"))

# =========================================================== (a) benchmark audit
a = canvas(.05, 5.18, 7.1, 1.42); title(a, "(a)", "Benchmark audit: one cell per record in the file", 146)
F = FL.assign(k=FL.duplicate * 10 + FL.y).sort_values(["duplicate", "y", "conflicting"], ascending=[True, True, False]).reset_index(drop=True)
P, NR = 11.4, 11; NC = int(np.ceil(len(F) / NR)); X0, Y0 = 2, 2
for i, r in F.iterrows():
    cx, cy = X0 + (i // NR) * P, Y0 + (NR - 1 - i % NR) * P
    a.add_patch(Rectangle((cx, cy), P - 1.5, P - 1.5, fc=CL[int(r.y)], alpha=1.0 if not r.duplicate else .32, ec="none"))
    if r.conflicting: a.add_patch(Circle((cx + (P - 1.5) / 2, cy + (P - 1.5) / 2), 1.35, fc=INK, ec="none"))
nu = int((F.duplicate == 0).sum()); cu = nu / NR; top = Y0 + NR * P + 1
for x1, x2, t in [(X0, X0 + cu * P - 1.5, f"{nu} unique rows"), (X0 + cu * P, X0 + NC * P - 1.5, f"{len(F) - nu} exact duplicates of another record")]:
    a.plot([x1, x1, x2, x2], [top, top + 3, top + 3, top], color=LINE, lw=.8); a.text((x1 + x2) / 2, top + 5, t, ha="center", va="bottom", fontsize=7.6)
xr = X0 + NC * P + 20
a.text(xr, 112, f"{A['unique_predictor_vectors']}", fontsize=13, fontweight="bold", va="center"); a.text(xr + 38, 112, "distinct predictor vectors", fontsize=7.8, va="center")
a.text(xr, 90, f"{A['predictor_vectors_with_conflicting_labels']}", fontsize=13, fontweight="bold", va="center")
a.text(xr + 38, 90, f"of them carry conflicting labels\n({A['rows_in_conflicting_vectors']} records, marked with a dot)", fontsize=7.8, va="center", linespacing=1.2)
rbox(a, xr - 6, 38, 288, 34, fc=PR, ec=RD, lw=.9)
a.text(xr + 2, 55, f"{100 * A['bayes_ceiling_accuracy_on_374']:.1f}%", fontsize=14, fontweight="bold", va="center")
a.text(xr + 68, 55, "accuracy ceiling: the most any classifier\ncan score on this file", fontsize=7.6, va="center", linespacing=1.2)
for k in range(3):
    xx = xr + [0, 58, 162][k]; a.add_patch(Rectangle((xx, 18), 8, 8, fc=CL[k], ec="none")); a.text(xx + 11, 22, CLN[k], fontsize=7.2, va="center")
a.add_patch(Rectangle((xr, 2), 8, 8, fc=MUTE, ec="none")); a.text(xr + 11, 6, "first occurrence", fontsize=7.2, va="center")
a.add_patch(Rectangle((xr + 84, 2), 8, 8, fc=MUTE, alpha=.32, ec="none")); a.text(xr + 95, 6, "duplicate", fontsize=7.2, va="center")
a.add_patch(Rectangle((xr + 142, 2), 8, 8, fc=MUTE, alpha=.32, ec="none")); a.add_patch(Circle((xr + 146, 6), 1.55, fc=INK, ec="none")); a.text(xr + 153, 6, "conflicting-label vector", fontsize=7.2, va="center")

# =========================================================== (b) the stack
b = canvas(.05, 2.68, 7.1, 2.2); title(b, "(b)", "Fold-isolated RF–XGBoost stack, fitted inside every training partition", 208)
BAND, GREY = "#f1f1ee", "#cfcfcf"
def band(x1, x2, head):
    rbox(b, x1, 36, x2 - x1, 162, fc=BAND, ec="none", r=5); b.text((x1 + x2) / 2, 188, head, ha="center", va="center", fontsize=7.8, fontweight="bold")
def block_arrow(x1, x2, y=112):
    b.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="simple,head_width=9,head_length=6,tail_width=3.6", fc="#9a9a9a", ec="none", shrinkA=0, shrinkB=0))
# 1 training partition
band(0, 94, "Training partition")
grid_cells(b, 19, 79, 6, 7, 9.5, 9.5, lambda r_, c_: CL[(r_ * 2 + 1) % 3] if c_ == 5 else GREY)
b.text(47, 70, "raw records with\ntheir labels", ha="center", va="top", fontsize=6.9, color=MUTE, linespacing=1.2)
block_arrow(95.5, 105.5)
# 2 inner split
band(107, 241, "Inner 5-fold grouped split"); strips(b, 118, 160, 112, 9.5, 14.5)
b.add_patch(Rectangle((118, 84), 8, 8, fc="#c9dcee", ec="none")); b.text(129, 88, "fit", fontsize=6.9, va="center", color=MUTE)
b.add_patch(Rectangle((149, 84), 8, 8, fc=OR, ec="none")); b.text(160, 88, "predict (out of fold)", fontsize=6.9, va="center", color=MUTE)
b.text(174, 72, "scaler and encoder refitted on\neach inner-training part", ha="center", va="top", fontsize=6.9, color=MUTE, linespacing=1.2)
block_arrow(242.5, 254.5)
# 3 base learners
band(256, 410, "Base learners")
for yb, name, sub, boost in [(118, "Random forest", "100 trees, depth 10", False), (52, "XGBoost", "800 rounds, depth 3", True)]:
    rbox(b, 264, yb, 138, 52, fc="white", ec=GN, lw=1.0)
    for k in range(3):
        tree(b, 271 + k * 16.5, yb + 17, 11.5, 18)
        if boost and k < 2: b.text(271 + k * 16.5 + 14, yb + 27, "+", ha="center", va="center", fontsize=5.5, color=GN)
    b.text(322, yb + 33, name, fontsize=7.8, fontweight="bold", va="center"); b.text(322, yb + 19, sub, fontsize=6.9, va="center", color=MUTE)
block_arrow(412, 424)
# 4 stacking matrix: [p_RF | p_XGB | x']
band(426, 572, "Stacking matrix"); cw, mx, my, nr = 9.2, 438, 78, 6
al = [[.95, .25, .2], [.2, .9, .3], [.3, .2, .95], [.25, .85, .35], [.9, .3, .25], [.2, .3, .9]]
al2 = [[.85, .35, .2], [.3, .8, .3], [.2, .3, .9], [.2, .9, .3], [.8, .4, .2], [.3, .2, .95]]
x_at = lambda k: mx + k * cw + (5 if k >= 3 else 0) + (5 if k >= 6 else 0)
for r_ in range(nr):
    for k in range(13):
        if k < 3: fc, alp = CL[k], al[r_][k]
        elif k < 6: fc, alp = CL[k - 3], al2[r_][k - 3]
        else: fc, alp = GREY, 1.0
        b.add_patch(Rectangle((x_at(k), my + (nr - 1 - r_) * cw), cw - 1, cw - 1, fc=fc, alpha=alp, ec="none"))
ytop = my + nr * cw + 2
for k1, k2, t in [(0, 3, r"$\mathbf{p}_{RF}$"), (3, 6, r"$\mathbf{p}_{XGB}$"), (6, 13, r"$\mathbf{x}^{\prime}$")]:
    b.plot([x_at(k1), x_at(k2 - 1) + cw - 1], [ytop, ytop], color=LINE, lw=.8); b.text((x_at(k1) + x_at(k2 - 1) + cw - 1) / 2, ytop + 3, t, ha="center", va="bottom", fontsize=8.6)
b.text((x_at(0) + x_at(5) + cw) / 2, my - 5, "out-of-fold class\nprobabilities", ha="center", va="top", fontsize=6.6, color=MUTE, linespacing=1.15)
b.text((x_at(6) + x_at(12) + cw) / 2, my - 5, "encoded\npredictors", ha="center", va="top", fontsize=6.6, color=MUTE, linespacing=1.15)
# pass-through route under the bands
xp = (x_at(6) + x_at(12) + cw) / 2
b.plot([47, 47, xp, xp], [44, 21, 21, 40], color=LINE, lw=1.0, ls=(0, (4, 2.5))); arrow(b, (xp, 40), (xp, 48))
b.text(236, 24, "encoded predictors, passed through unchanged", ha="center", va="bottom", fontsize=6.9, color=MUTE)
block_arrow(574, 586)
# 5 meta-learner and output
band(588, 710, "Meta-learner")
rbox(b, 598, 96, 102, 44, fc=PO, ec=OR, lw=1.0); b.text(649, 118, "Multinomial\nlogistic regression", ha="center", va="center", fontsize=7.6, linespacing=1.2)
arrow(b, (649, 96), (649, 84))
for k, wv in enumerate([40, 15, 8]): b.add_patch(Rectangle((629, 72 - k * 8.5), wv, 6.5, fc=CL[k], ec="none"))
b.text(649, 46, "class probabilities", ha="center", va="top", fontsize=6.9, color=MUTE)
b.text(710, 2, "For prediction, both base learners and the transformer are refitted on the full training partition.", ha="right", va="bottom", fontsize=6.8, color=MUTE, style="italic")

# =========================================================== (c) evaluation protocol
c = canvas(.05, .16, 4.15, 2.38); title(c, "(c)", "Evaluation protocol", 222)
for k in range(3, 0, -1): c.add_patch(Rectangle((6 + 4 * k, 98 - 4 * k), 148, 96, fc="white", ec="#bdbdbd", lw=.7))
c.add_patch(Rectangle((6, 98), 148, 96, fc="white", ec=LINE, lw=.9)); strips(c, 15, 172, 130, 10, 16)
c.text(80, 201, "10 repeats × 5 grouped folds", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
c.add_patch(Rectangle((6, 66), 9, 9, fc="#c9dcee", ec="none")); c.text(19, 70.5, "training folds", fontsize=7.2, va="center")
c.add_patch(Rectangle((84, 66), 9, 9, fc=OR, ec="none")); c.text(97, 70.5, "test fold, scored once", fontsize=7.2, va="center")
c.text(6, 52, f"Records with an identical predictor\nvector always share a fold ({A['unique_predictor_vectors']} groups),\nso no vector is both trained and tested.", fontsize=7.2, va="top", linespacing=1.25)
items = [("Nested tuning", "hyperparameters chosen by an inner 3-fold\ngrouped search on the training partition only", GN),
         ("Unique rows", f"the {A['unique_rows']} distinct records only, as a\nsensitivity analysis", OR),
         ("Single 80:20 split", f"re-executed; {A['orig_split_test_rows_with_identical_train_vector']} of {A['orig_split_test_rows']} test records repeat\na training predictor vector", GY),
         ("Same folds for every model", f"{NCOMP} comparators, paired tests, calibration,\nSHAP and LIME audits", BL)]
for k, (h, t, col) in enumerate(items):
    yy = 160 - k * 54; rbox(c, 186, yy, 226, 46, fc=SOFT, ec="none", r=3); c.add_patch(Rectangle((186, yy), 3.5, 46, fc=col, ec="none"))
    c.text(196, yy + 35, h, fontsize=7.8, fontweight="bold", va="center"); c.text(196, yy + 16, t, fontsize=7.0, va="center", linespacing=1.2, color="#333333")

# =========================================================== (d) accuracy under each treatment
d = fig.add_axes([5.23 / FW, .78 / FH, 1.87 / FW, 1.5 / FH])
mods = [("LogReg", "Logistic regression", GY, "s"), ("SoftVote(RF+XGB)", "Soft voting", BL, "o"), ("Stack(RF+XGB)", "Stacking", RD, "D")]
treat = [("hold", "Single 80:20\nsplit"), ("naive", "Plain stratified\nCV"), ("grouped", "Grouped CV\n(primary)"), ("dedup", "Unique rows\nonly")]
for i, (tk, tl) in enumerate(treat):
    yv = len(treat) - 1 - i
    if tk == "grouped": d.axhspan(yv - .5, yv + .5, color=BL, alpha=.08, lw=0)
    for j, (m, ml, col, mk) in enumerate(mods):
        if tk == "hold": mu, lo, hi = OS[m]["test_acc"], *OS[m]["clopper_pearson_95"]
        else: r = S.loc[(tk, m)]; mu, lo, hi = r["acc"], r["acc_bs_lo"], r["acc_bs_hi"]
        yy = yv + (1 - j) * .24
        d.plot([100 * lo, 100 * hi], [yy, yy], color=col, lw=1.3, solid_capstyle="butt"); d.scatter([100 * mu], [yy], s=24, marker=mk, color=col, edgecolor="white", lw=.7, zorder=3)
        if m.startswith("Stack"): d.text(100 * lo - 1.5, yy, f"{100 * mu:.1f}", fontsize=7.6, ha="right", va="center", fontweight="bold")
cl = 100 * A["bayes_ceiling_accuracy_on_374"]; d.axvline(cl, color=INK, ls=(0, (4, 2.5)), lw=.8)
d.annotate(f"ceiling {cl:.1f}%", xy=(cl, len(treat) - .5), xytext=(0, 2), textcoords="offset points", fontsize=7, ha="center", va="bottom", annotation_clip=False)
d.set_yticks(range(len(treat))); d.set_yticklabels([t for _, t in treat][::-1], fontsize=7.6); d.set_xlim(58, 101); d.set_ylim(-.55, len(treat) - .5)
d.set_xlabel("Accuracy (%)", fontsize=8); d.tick_params(axis="x", labelsize=7.6); d.tick_params(axis="y", length=0); grid(d, "x"); d.spines["left"].set_visible(False)
d.get_yticklabels()[1].set_fontweight("bold")
fig.text(4.42 / FW, (.16 + 2.228) / FH, "(d)", fontweight="bold", fontsize=11, ha="left", va="bottom"); fig.text(4.42 / FW + 23 / 100 / FW, (.16 + 2.236) / FH, "Accuracy under each treatment", fontsize=9.2, ha="left", va="bottom")
d.legend(handles=[Line2D([], [], marker=mk, color=col, mec="white", mew=.7, ms=5, lw=1.3, label=ml) for _, ml, col, mk in mods], frameon=False, ncol=3, loc="upper center",
         bbox_to_anchor=(.28, -.25), fontsize=7.2, columnspacing=.9, handlelength=1.5, handletextpad=.4)
fig.savefig(O + "fig1_overview.pdf"); fig.savefig(O + "fig1_overview.png", dpi=300); plt.close(fig); print("done")
