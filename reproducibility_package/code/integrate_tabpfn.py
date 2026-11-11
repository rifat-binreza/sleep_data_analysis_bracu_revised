"""Fold the Kaggle TabPFN run into the manuscript.

1. put tabpfn_oof_grouped.csv, tabpfn_oof_dedup.csv and tabpfn_meta.json (downloaded from Kaggle) into results/
2. python code/integrate_tabpfn.py
It computes the same quantities, in the same way, as stats_calib.py does for every other model (cluster-bootstrap
interval, macro-F1, AUC, Brier, log-loss, ECE, paired comparison with the stack), then writes
  results/tabpfn_summary.csv, results/tabpfn_paired.csv
  ../tabpfn_numbers.tex      table rows and sentences picked up automatically by Final_elsevier_revised.tex
and redraws the figures that list all models.  Needs data/sleep.csv (see data/README.txt).
"""
import os, sys, json, subprocess, numpy as np, pandas as pd
from math import comb
from scipy import stats
from sklearn.metrics import f1_score, log_loss, roc_auc_score, balanced_accuracy_score
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results", ""); ROOT = os.path.join(HERE, "..", "..", "")
CSV = os.environ.get("SLEEP_CSV", os.path.join(HERE, "..", "data", "sleep.csv"))
MODEL = os.environ.get("INTEGRATE_MODEL", "TabPFN"); SRC = os.environ.get("INTEGRATE_SRC", "tabpfn_oof_{}.csv")   # overridable only to validate this script on a released model
FEATS = ["Gender", "Age", "Occupation", "Sleep Duration", "Quality of Sleep", "Physical Activity Level", "Stress Level", "BMI Category", "Heart Rate", "Daily Steps", "Systolic_BP", "Diastolic_BP"]
PC = ["p_insomnia", "p_none", "p_apnea"]; STACK = "Stack(RF+XGB)"

def load(proto):
    df = pd.read_csv(CSV, keep_default_na=False, na_values=[""]).drop(columns=["Person ID"]); df["BMI Category"] = df["BMI Category"].replace({"Normal Weight": "Normal"})
    bp = df["Blood Pressure"].str.split("/", expand=True); df["Systolic_BP"] = bp[0].astype(float); df["Diastolic_BP"] = bp[1].astype(float)
    y = df["Sleep Disorder"].map({"Insomnia": 0, "None": 1, "Sleep Apnea": 2}).values; X = df[FEATS].reset_index(drop=True)
    if proto == "dedup":
        keep = ~X.assign(y=y).duplicated().values; X, y = X[keep].reset_index(drop=True), y[keep]
    return y, pd.factorize(X.astype(str).agg("|".join, axis=1))[0]
def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    return min(1.0, sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n * 2)
def ece(P, y, bins=5):
    conf = P.max(1); hit = (P.argmax(1) == y).astype(float); edges = np.quantile(conf, np.linspace(0, 1, bins + 1)); edges[-1] += 1e-9; e = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum(): e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return e
def fold_metrics(d):
    y, P = d.y.values, d[PC].values; P = P / P.sum(1, keepdims=True); Y = np.eye(3)[y]
    try: auc = roc_auc_score(y, P, multi_class="ovr", average="macro", labels=[0, 1, 2])
    except ValueError: auc = np.nan
    return pd.Series(dict(acc=(d.pred.values == y).mean(), macro_f1=f1_score(y, d.pred.values, average="macro", labels=[0, 1, 2], zero_division=0),
                          bal_acc=balanced_accuracy_score(y, d.pred.values), logloss=log_loss(y, P, labels=[0, 1, 2]), brier=((P - Y) ** 2).sum(1).mean(), auc=auc))

rng = np.random.default_rng(0); summ, pair = [], []
for proto in ["grouped", "dedup"]:
    y, g = load(proto); pm = pd.read_csv(R + SRC.format(proto)); pm = pm[pm.model == MODEL]
    assert len(pm) == 10 * len(y) and (pm.y.values == y[pm.row.values]).all(), "TabPFN file does not match the dataset rows"
    ps = pd.read_csv(R + f"oof_predictions_{proto}.csv"); ps = ps[ps.model == STACK]
    chk = pm.merge(ps, on=["repeat", "row"], suffixes=("", "_s")); assert len(chk) == len(pm) and (chk.fold == chk.fold_s).all(), "folds differ from the released split"
    fo = pm.groupby(["repeat", "fold"])[["y", "pred"] + PC].apply(fold_metrics).reset_index(); rep = fo.groupby("repeat")[["acc", "macro_f1", "bal_acc", "brier", "logloss", "auc"]].mean()
    d = dict(protocol=proto, model="TabPFN")
    for k in rep.columns:
        mu, sd = rep[k].mean(), rep[k].std(ddof=1); h = stats.t.ppf(.975, len(rep) - 1) * sd / np.sqrt(len(rep)); d[k] = mu; d[k + "_sd_rep"] = sd; d[k + "_lo"] = mu - h; d[k + "_hi"] = mu + h
    d["acc_sd_fold"] = fo.acc.std(ddof=1)
    cm = pm.assign(c=(pm.pred == pm.y).astype(int)).pivot(index="row", columns="repeat", values="c").sort_index(); rowacc = cm.mean(axis=1).values
    ug, inv = np.unique(g, return_inverse=True); ib = [np.where(inv == k)[0] for k in range(len(ug))]
    bsa = np.array([rowacc[np.concatenate([ib[k] for k in rng.integers(0, len(ug), len(ug))])].mean() for _ in range(5000)])
    d["acc_bs_lo"], d["acc_bs_hi"] = float(np.percentile(bsa, 2.5)), float(np.percentile(bsa, 97.5))
    d["ece5"] = float(np.mean([ece(pp.sort_values("row")[PC].values, pp.sort_values("row").y.values) for _, pp in pm.groupby("repeat")])); summ.append(d)
    # paired comparison with the stack (stack minus TabPFN)
    cs = ps.assign(c=(ps.pred == ps.y).astype(int)).pivot(index="row", columns="repeat", values="c").sort_index(); row_d = (cs.values - cm.values).mean(1)
    bs = np.array([row_d[np.concatenate([ib[k] for k in rng.integers(0, len(ug), len(ug))])].mean() for _ in range(20000)])
    fs = ps.assign(c=(ps.pred == ps.y)).groupby(["repeat", "fold"]).c.mean(); fd = (fs - fo.set_index(["repeat", "fold"]).acc).values; J = len(fd); s2 = fd.var(ddof=1)
    tnb = fd.mean() / np.sqrt((1 / J + 4.0) * s2) if s2 > 0 else 0.0; pnb = 2 * (1 - stats.t.cdf(abs(tnb), J - 1))
    pvals = [mcnemar_exact(int(((cs[r] == 1) & (cm[r] == 0)).sum()), int(((cs[r] == 0) & (cm[r] == 1)).sum())) for r in cs.columns]
    pair.append(dict(protocol=proto, comparator="TabPFN", n=len(y), mean_acc_diff_pp=100 * row_d.mean(), boot_lo_pp=100 * np.percentile(bs, 2.5), boot_hi_pp=100 * np.percentile(bs, 97.5),
                     nb_t=tnb, nb_p=pnb, folds_stack_better=int((fd > 1e-12).sum()), folds_tie=int((abs(fd) <= 1e-12).sum()), folds_comp_better=int((fd < -1e-12).sum()),
                     mcnemar_p_median=float(np.median(pvals)), repeats_p_lt_05=int(sum(p < .05 for p in pvals))))
Sm, Pr = pd.DataFrame(summ), pd.DataFrame(pair)
if MODEL != "TabPFN":                                   # validation mode: print and stop, write nothing
    print(Sm[["protocol", "acc", "acc_bs_lo", "acc_bs_hi", "macro_f1", "auc", "brier", "logloss", "ece5"]].round(4).to_string(index=False)); print(Pr.round(4).to_string(index=False)); sys.exit(0)
Sm.to_csv(R + "tabpfn_summary.csv", index=False); Pr.to_csv(R + "tabpfn_paired.csv", index=False)
meta = json.load(open(R + "tabpfn_meta.json")) if os.path.exists(R + "tabpfn_meta.json") else {}
pc = lambda v: f"{100 * v:.1f}"; f2 = lambda v: f"{v:.2f}"; f3 = lambda v: f"{v:.3f}"; sg = lambda v: f"{v:+.1f}"
pval = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
S_ = Sm.set_index("protocol"); P_ = Pr.set_index("protocol"); G, D, PG, PD = S_.loc["grouped"], S_.loc["dedup"], P_.loc["grouped"], P_.loc["dedup"]
prow = lambda q: f"TabPFN (v2) & {sg(q.mean_acc_diff_pp)} [{sg(q.boot_lo_pp)}, {sg(q.boot_hi_pp)}] & {q.folds_stack_better}/{q.folds_tie}/{q.folds_comp_better} & {pval(q.mcnemar_p_median)} & {q.repeats_p_lt_05}/10 & {pval(q.nb_p)}\\\\"
ver = meta.get("tabpfn_version", "")
tex = f"""% written by code/integrate_tabpfn.py from the Kaggle TabPFN run - do not edit by hand
\\renewcommand{{\\NBase}}{{thirteen}}
\\newcommand{{\\AccTabG}}{{{pc(G.acc)}}}\\newcommand{{\\CIloTabG}}{{{pc(G.acc_bs_lo)}}}\\newcommand{{\\CIhiTabG}}{{{pc(G.acc_bs_hi)}}}
\\newcommand{{\\AccTabD}}{{{pc(D.acc)}}}\\newcommand{{\\CIloTabD}}{{{pc(D.acc_bs_lo)}}}\\newcommand{{\\CIhiTabD}}{{{pc(D.acc_bs_hi)}}}
\\newcommand{{\\DiffTabG}}{{{sg(PG.mean_acc_diff_pp)}}}\\newcommand{{\\DiffLoTabG}}{{{sg(PG.boot_lo_pp)}}}\\newcommand{{\\DiffHiTabG}}{{{sg(PG.boot_hi_pp)}}}
\\newcommand{{\\DiffTabD}}{{{sg(PD.mean_acc_diff_pp)}}}\\newcommand{{\\DiffLoTabD}}{{{sg(PD.boot_lo_pp)}}}\\newcommand{{\\DiffHiTabD}}{{{sg(PD.boot_hi_pp)}}}
\\renewcommand{{\\TabPFNMethods}}{{The pretrained tabular foundation model TabPFN \\cite{{hollmann2025}} (version~2 weights{', package version ' + ver if ver else ''}) was evaluated on the same folds with its default settings and no tuning; categorical predictors were integer-coded on each training partition and numerical predictors were passed unscaled, as the model performs its own preprocessing.}}
\\renewcommand{{\\TabPFNResults}}{{TabPFN reached \\AccTabG\\% (95\\% CI \\CIloTabG--\\CIhiTabG) under grouped cross-validation and \\AccTabD\\% (\\CIloTabD--\\CIhiTabD) on the unique rows; the stack differed from it by $\\DiffTabG$~pp (95\\% CI $\\DiffLoTabG$ to $\\DiffHiTabG$) and $\\DiffTabD$~pp ($\\DiffLoTabD$ to $\\DiffHiTabD$), respectively (Table~\\ref{{tab:cvmain}}; the paired analysis is in Section~\\ref{{sec:paired}}).}}
\\renewcommand{{\\TabPFNLimitation}}{{}}
\\renewcommand{{\\TabPFNRowMain}}{{TabPFN (v2) & {pc(G.acc)} [{pc(G.acc_bs_lo)}, {pc(G.acc_bs_hi)}] & {f2(G.macro_f1)} & {f2(G.auc)} & {f3(G.brier)} & {f3(G.logloss)} & {f3(G.ece5)}\\\\}}
\\renewcommand{{\\TabPFNRowPairedG}}{{{prow(PG)}}}
\\renewcommand{{\\TabPFNRowPairedD}}{{{prow(PD)}}}
"""
open(ROOT + "tabpfn_numbers.tex", "w").write(tex); print(tex)
env = dict(os.environ, SLEEP_CSV=CSV)
for cmd in (["code/perclass_diffs.py"], ["code/conflict_errors.py"], ["figure_scripts/fig_overview.py"], ["figure_scripts/figs_results.py"]):
    subprocess.check_call([sys.executable, os.path.join(HERE, "..", *cmd[0].split("/"))], env=env, stdout=subprocess.DEVNULL)
print("Figures redrawn. Recompile Final_elsevier_revised.tex and Response_to_Reviewers.tex.")
