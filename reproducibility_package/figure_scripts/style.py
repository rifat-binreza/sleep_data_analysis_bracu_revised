"""Shared figure style for the revision (same fonts/palette as figs_q1.py)."""
import os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "results", ""); O = os.path.join(H, "..", "..", "figs", "")
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8, "axes.linewidth": .8,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.major.width": .8, "ytick.major.width": .8, "pdf.fonttype": 42, "axes.axisbelow": True})
BL, OR, GN, VM, SK, GY, RD = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#7f7f7f", "#D55E00"   # Okabe-Ito
INK = "#222222"
CLASS_COL = {0: BL, 1: OR, 2: GN}; CLASS_NAME = {0: "Insomnia", 1: "No recorded\ndisorder", 2: "Sleep apnea"}
NAMES = {"Majority": "Majority class", "Tree(d=3)": "Decision tree (depth 3)", "LogReg": "Logistic regression", "SVM(rbf)": "SVM (RBF)",
         "MLP": "MLP (32,16)", "FT-Transformer": "FT-Transformer", "TabPFN": "TabPFN (v2)", "RF": "Random forest", "ExtraTrees": "Extra trees",
         "HistGB": "HistGradientBoosting", "LightGBM": "LightGBM", "CatBoost": "CatBoost", "XGB": "XGBoost",
         "SoftVote(RF+XGB)": "Soft voting (RF+XGB)", "Stack(RF+XGB)": "Stacking (RF+XGB)"}
ORDER = list(NAMES)
def tag(ax, s, x=-0.02, y=1.04): ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", fontsize=11, ha="right", va="bottom")
def grid(ax, axis="x"): ax.grid(axis=axis, color="#e3e3e3", lw=.7)
def save(fig, name):
    fig.savefig(O + name + ".pdf", bbox_inches="tight"); fig.savefig(O + name + ".png", bbox_inches="tight", dpi=300); plt.close(fig)
def ptitle(ax, lab, ttl):
    """Bold panel tag followed by the panel title, left-aligned above the axes."""
    ax.text(0, 1.03, lab, transform=ax.transAxes, fontweight="bold", fontsize=11, ha="left", va="bottom")
    ax.annotate(ttl, xy=(0, 1.035), xycoords="axes fraction", xytext=(21, 0), textcoords="offset points", fontsize=9, ha="left", va="bottom")
