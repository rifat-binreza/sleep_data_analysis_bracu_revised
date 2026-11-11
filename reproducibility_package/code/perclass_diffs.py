"""Paired differences between the stack and every comparator, overall and by reference class.
For each protocol (grouped CV on 374 records; CV on the 132 unique rows) and comparator:
  accuracy difference        taken from paired_vs_stack.csv / tabpfn_paired.csv (same values as the paired table)
  per-class recall difference computed here the same way: per-record correctness averaged over the 10 repeats,
                              stack minus comparator, 95% interval by cluster bootstrap over predictor vectors.
Writes results/perclass_diffs.csv and perclass_numbers.tex.  usage: python perclass_diffs.py
"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results", "")
CSV = os.environ.get("SLEEP_CSV", os.path.join(HERE, "..", "data", "sleep.csv")); STACK = "Stack(RF+XGB)"; B = 10000
FEATS = ["Gender", "Age", "Occupation", "Sleep Duration", "Quality of Sleep", "Physical Activity Level", "Stress Level", "BMI Category", "Heart Rate", "Daily Steps", "Systolic_BP", "Diastolic_BP"]
def load(proto):
    df = pd.read_csv(CSV, keep_default_na=False, na_values=[""]).drop(columns=["Person ID"]); df["BMI Category"] = df["BMI Category"].replace({"Normal Weight": "Normal"})
    bp = df["Blood Pressure"].str.split("/", expand=True); df["Systolic_BP"] = bp[0].astype(float); df["Diastolic_BP"] = bp[1].astype(float)
    y = df["Sleep Disorder"].map({"Insomnia": 0, "None": 1, "Sleep Apnea": 2}).values; X = df[FEATS].reset_index(drop=True)
    if proto == "dedup":
        keep = ~X.assign(y=y).duplicated().values; X, y = X[keep].reset_index(drop=True), y[keep]
    return y, pd.factorize(X.astype(str).agg("|".join, axis=1))[0]
rng = np.random.default_rng(20261111); rows = []
for proto in ["grouped", "dedup"]:
    y, g = load(proto); p = pd.read_csv(RES + f"oof_predictions_{proto}.csv")
    if os.path.exists(RES + f"tabpfn_oof_{proto}.csv"): p = pd.concat([p, pd.read_csv(RES + f"tabpfn_oof_{proto}.csv")], ignore_index=True)
    assert (p.y.values == y[p.row.values]).all()
    acc = pd.read_csv(RES + "paired_vs_stack.csv")
    if os.path.exists(RES + "tabpfn_paired.csv"): acc = pd.concat([acc, pd.read_csv(RES + "tabpfn_paired.csv")], ignore_index=True)
    acc = acc[acc.protocol == proto].set_index("comparator")
    C = {m: q.assign(c=(q.y == q.pred).astype(float)).pivot(index="row", columns="repeat", values="c").sort_index().mean(axis=1).values for m, q in p.groupby("model")}
    ug = np.unique(g); idx = [np.where(g == k)[0] for k in ug]
    draws = [np.concatenate([idx[k] for k in rng.integers(0, len(ug), len(ug))]) for _ in range(B)]       # same resamples for every comparator
    for m in C:
        if m in (STACK, "Majority"): continue
        d = 100 * (C[STACK] - C[m]); a = acc.loc[m]
        rows.append(dict(protocol=proto, comparator=m, metric="Accuracy", diff=a.mean_acc_diff_pp, lo=a.boot_lo_pp, hi=a.boot_hi_pp))
        for k, nm in enumerate(["Insomnia recall", "No-disorder recall", "Sleep-apnea recall"]):
            bs = np.array([d[ii][y[ii] == k].mean() for ii in draws if (y[ii] == k).any()]); lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(dict(protocol=proto, comparator=m, metric=nm, diff=d[y == k].mean(), lo=lo, hi=hi))
D = pd.DataFrame(rows); # an interval counts as excluding zero only at the reported precision (0.1 pp)
D["excl0"] = (D.lo.round(1) > 0) | (D.hi.round(1) < 0); D.to_csv(RES + "perclass_diffs.csv", index=False)
n, k = len(D), int(D.excl0.sum()); kp = int((D.excl0 & (D["diff"] > 0)).sum()); kc = D[D.metric != "Accuracy"]
G = D[D.protocol == "grouped"]; Gn = G[(G.metric == "No-disorder recall") & (G["diff"] < 0)]; Ga = G[(G.metric == "Sleep-apnea recall") & G.excl0 & (G.comparator != "Tree(d=3)")]
tex = (f"\\newcommand{{\\PCn}}{{{n}}}\\newcommand{{\\PCexcl}}{{{k}}}\\newcommand{{\\PCexclG}}{{{int(G.excl0.sum())}}}\\newcommand{{\\PCexclD}}{{{k - int(G.excl0.sum())}}}\n"
       f"\\newcommand{{\\PCnoneN}}{{{len(Gn)}}}\\newcommand{{\\PCnoneMax}}{{{-Gn['diff'].min():.1f}}}\\newcommand{{\\PCapneaLo}}{{{Ga['diff'].min():.1f}}}\\newcommand{{\\PCapneaHi}}{{{Ga['diff'].max():.1f}}}\n")
for dst in [RES + "perclass_numbers.tex", os.path.join(HERE, "..", "..", "paper_tables", "perclass_numbers.tex")]: open(dst, "w").write(tex)
print(tex); print(D[D.excl0].round(2).to_string(index=False)); print(D.groupby(["protocol", "metric"])["diff"].agg(["min", "max"]).round(2))
