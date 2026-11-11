import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "../results/"; O = "../figs/"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
BL, OR, GN, RD, GY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#7f7f7f"
d = pd.read_csv(R + "ablation_stack_all.csv"); order = ["Full stack (RF+XGB+passthrough)", "Stack without RF", "Stack without XGBoost", "Stack without passthrough", "Soft voting (RF+XGB)", "RF only", "XGBoost only"]
lab = ["Full stack\n(RF+XGB+pass-through)", "Stack\nwithout RF", "Stack without\nXGBoost", "Stack without\npass-through", "Soft voting\n(RF+XGB)", "RF only", "XGBoost\nonly"]
rep = d.groupby(["model", "repeat"])[["acc", "macro_f1", "auc"]].mean().reset_index(); fig, ax = plt.subplots(figsize=(7.4, 3.3)); w = .26
for j, (k, n, c) in enumerate([("acc", "Accuracy", BL), ("macro_f1", "Macro-F1", OR), ("auc", "Macro ROC–AUC", GN)]):
    mu = np.array([rep[rep.model == m][k].mean() for m in order]); sd = np.array([rep[rep.model == m][k].std(ddof=1) for m in order])
    ax.bar(np.arange(7) + (j - 1) * w, mu, w, yerr=1.96 * sd / np.sqrt(10), color=c, label=n, capsize=2, error_kw=dict(lw=.8), edgecolor="white")
ax.set_xticks(range(7)); ax.set_xticklabels(lab, fontsize=7.6); ax.set_ylim(.7, 1.0); ax.set_ylabel("Grouped 10$\\times$5 CV score"); ax.grid(axis="y", color="#e6e6e6", lw=.7); ax.axhline(.5865, color=GY)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.13), fontsize=8)
plt.savefig(O + "output.png", dpi=220, bbox_inches="tight"); plt.close()
s = pd.read_csv(R + "seed_sweep.csv"); fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True, gridspec_kw=dict(wspace=.08))
for ax, m, t in zip(axs, ["Soft voting (RF+XGB)", "Stack (RF+XGB)"], ["(a) RF–XGBoost soft voting", "(b) RF–XGBoost stack"]):
    q = s[s.model == m]; ax.fill_between(q.seed, q.train_lo, q.train_hi, color=RD, alpha=.18, lw=0); ax.fill_between(q.seed, q.test_lo, q.test_hi, color=BL, alpha=.18, lw=0)
    ax.plot(q.seed, q.train_acc, "o-", color=RD, ms=3.5, lw=1.3, label="Training accuracy (n = 299)"); ax.plot(q.seed, q.test_acc, "o-", color=BL, ms=3.5, lw=1.3, label="Test accuracy (n = 75)")
    ax.set_title(t, fontsize=9, loc="left"); ax.set_xlabel("Random split seed"); ax.set_xticks(range(50, 651, 100)); ax.grid(color="#e6e6e6", lw=.7); ax.set_ylim(.75, 1.02)
axs[0].set_ylabel("Accuracy"); axs[0].legend(frameon=False, loc="lower left", fontsize=7.5)
plt.savefig(O + "12.png", dpi=220, bbox_inches="tight"); plt.close()
print(rep.groupby("model")[["acc", "macro_f1", "auc"]].agg(["mean", "std"]).round(4).to_string())
