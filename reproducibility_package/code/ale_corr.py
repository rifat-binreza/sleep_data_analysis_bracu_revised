import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from common import *
R = "../results/"; O = "../figs/"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 9, "pdf.fonttype": 42})
X, y = load(); tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=42)
Xd = X.iloc[tr].reset_index(drop=True); yd = y[tr]; st = OOFStack(seed=42).fit(Xd, yd)
f = lambda D: st.predict_proba(D)[:, 0]
a, b, K = "Sleep Duration", "Quality of Sleep", 6
def edges(v): e = np.unique(np.quantile(v, np.linspace(0, 1, K + 1))); return e
ea, eb = edges(Xd[a].values), edges(Xd[b].values); na, nb = len(ea) - 1, len(eb) - 1
ia = np.clip(np.searchsorted(ea, Xd[a].values, side="right") - 1, 0, na - 1); ib = np.clip(np.searchsorted(eb, Xd[b].values, side="right") - 1, 0, nb - 1)
D = np.zeros((na, nb)); N = np.zeros((na, nb))
for i in range(na):
    for j in range(nb):
        m = (ia == i) & (ib == j)
        if not m.any(): continue
        Z = Xd[m]; 
        def ev(x, y_): T = Z.copy(); T[a] = x; T[b] = y_; return f(T)
        D[i, j] = np.mean(ev(ea[i + 1], eb[j + 1]) - ev(ea[i], eb[j + 1]) - ev(ea[i + 1], eb[j]) + ev(ea[i], eb[j])); N[i, j] = m.sum()
A2 = np.cumsum(np.cumsum(D, 0), 1); A2 = np.pad(A2, ((1, 0), (1, 0)))        # accumulated at grid corners
w = np.pad(N, ((1, 0), (1, 0))); cen = lambda M, W: M - (M * W).sum() / max(W.sum(), 1)
Wc = np.zeros_like(A2)
for i in range(na + 1):
    for j in range(nb + 1): Wc[i, j] = N[max(i - 1, 0):i + 1, max(j - 1, 0):j + 1].sum()
A2 = cen(A2, Wc)
rho = Xd[NUM].corr(method="spearman")
fig = plt.figure(figsize=(12.5, 5.2)); fig.suptitle("Feature relationships and stacking-model local effects", fontweight="bold", fontsize=12)
ax1 = fig.add_subplot(1, 2, 1); sns.heatmap(rho, ax=ax1, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws=dict(size=7), linewidths=.5, cbar_kws=dict(label="Spearman $r_s$", shrink=.8))
ax1.set_xticklabels([n.replace("Physical Activity Level", "Phys. Activity") for n in NUM], rotation=45, ha="right", fontsize=8); ax1.set_yticklabels([n.replace("Physical Activity Level", "Phys. Activity") for n in NUM], rotation=0, fontsize=8); ax1.set_title("(a) Spearman correlation coefficient", fontweight="bold")
ax2 = fig.add_subplot(1, 2, 2, projection="3d"); GA, GB = np.meshgrid(ea, eb, indexing="ij")
surf = ax2.plot_surface(GA, GB, A2, cmap="coolwarm", edgecolor="white", linewidth=.4, alpha=.95); ax2.set_xlabel(a, labelpad=8); ax2.set_ylabel(b, labelpad=8); ax2.set_zlabel("Centred ALE")
ax2.view_init(25, 235); ax2.set_title("(b) Accumulated Local Effects: Insomnia", fontweight="bold"); fig.colorbar(surf, ax=ax2, shrink=.6, pad=.12, label="ALE for P(Insomnia)")
plt.savefig(O + "feature_relationships_ale.png", dpi=200, bbox_inches="tight")
json.dump(dict(spearman={f"{p}|{q}": float(rho.loc[p, q]) for p in NUM for q in NUM if p < q}, ale_min=float(A2.min()), ale_max=float(A2.max()), grid_edges_a=ea.tolist(), grid_edges_b=eb.tolist()), open(R + "ale_corr.json", "w"), indent=1)
print(ea, eb, A2.min(), A2.max()); r = rho; print(r.loc["Sleep Duration", "Quality of Sleep"], r.loc["Stress Level", "Sleep Duration"], r.loc["Stress Level", "Quality of Sleep"], r.loc["Systolic_BP", "Diastolic_BP"], r.loc["Physical Activity Level", "Daily Steps"])
