"""Local LIME explanation for the audited record (Person ID 244), stack vs soft voting.
Reads the weights written by code/holdout_figures.py (results/holdout_figures.json)."""
import json, re
from style import *

L = json.load(open(R + "holdout_figures.json"))["lime_fig"]


def pretty(s):
    s = re.sub(r'(\d+)\.00\b', r'\1', s)
    s = s.replace('<=', '≤').replace('_BP', ' BP').replace('BMI Category', 'BMI category')
    s = re.sub(r'\s*=\s*', ' = ', s)
    return s


fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.15), gridspec_kw=dict(wspace=0.95))
lim = max(abs(v) for k in L for _, v in L[k]["w"]) * 1.12
for ax, k, lab, ttl in zip(axs, ["stack", "vote"], ["(a)", "(b)"], ["Stacking classifier", "Soft voting (RF+XGB)"]):
    w = L[k]["w"][::-1]
    vals = [v for _, v in w]
    ax.barh(range(len(w)), vals, color=[GN if v > 0 else RD for v in vals], height=.68, edgecolor="white", lw=.6)
    ax.set_yticks(range(len(w))); ax.set_yticklabels([pretty(n) for n, _ in w], fontsize=8.5)
    ax.axvline(0, color=INK, lw=.8); grid(ax)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Local surrogate weight (class: insomnia)")
    ax.tick_params(axis="y", length=0)
    ax.text(0, 1.10, lab, transform=ax.transAxes, fontweight="bold", fontsize=11, ha="left", va="bottom")
    ax.annotate(ttl, xy=(0, 1.105), xycoords="axes fraction", xytext=(21, 0), textcoords="offset points", fontsize=9.5, ha="left", va="bottom")
    ax.text(0, 1.03, "P(insomnia) = %.2f;  surrogate $R^2$ = %.2f" % (L[k]["p"][0], L[k]["r2"]),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="#555555")
h = [plt.Rectangle((0, 0), 1, 1, color=GN), plt.Rectangle((0, 0), 1, 1, color=RD)]
fig.legend(h, ["raises the insomnia score", "lowers the insomnia score"], loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.12), fontsize=8.5)
save(fig, "fig_lime_local")
