"""Collect the numbers shown on the interactive page (docs/index.html) into docs/data.js.
Reads only the released result files.  usage: python docs/build_data.py
"""
import os, json, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "reproducibility_package", "results", "")
NAMES = {"Majority": "Majority class", "Tree(d=3)": "Decision tree (depth 3)", "LogReg": "Logistic regression", "SVM(rbf)": "SVM (RBF)", "MLP": "MLP (32,16)",
         "FT-Transformer": "FT-Transformer", "TabPFN": "TabPFN (v2)", "RF": "Random forest", "ExtraTrees": "Extra trees", "HistGB": "HistGradientBoosting",
         "LightGBM": "LightGBM", "CatBoost": "CatBoost", "XGB": "XGBoost", "SoftVote(RF+XGB)": "Soft voting (RF+XGB)", "Stack(RF+XGB)": "Stacking (RF+XGB)"}
FAMILY = {"Majority": "Reference", "Tree(d=3)": "Simple models", "LogReg": "Simple models", "SVM(rbf)": "Simple models", "MLP": "Neural and pretrained", "FT-Transformer": "Neural and pretrained",
          "TabPFN": "Neural and pretrained", "RF": "Tree ensembles", "ExtraTrees": "Tree ensembles", "HistGB": "Tree ensembles", "LightGBM": "Tree ensembles", "CatBoost": "Tree ensembles",
          "XGB": "Tree ensembles", "SoftVote(RF+XGB)": "Same base learners", "Stack(RF+XGB)": "Evaluated model"}
r4 = lambda v: None if v is None or pd.isna(v) else round(float(v), 4)
S = pd.concat([pd.read_csv(R + "summary_all_protocols.csv"), pd.read_csv(R + "tabpfn_summary.csv")], ignore_index=True)
models = [dict(protocol=r.protocol, id=r.model, name=NAMES[r.model], family=FAMILY[r.model], acc=r4(100 * r.acc), lo=r4(100 * r.acc_bs_lo), hi=r4(100 * r.acc_bs_hi),
               f1=r4(r.macro_f1), auc=r4(getattr(r, "auc", None)), brier=r4(r.brier), logloss=r4(r.logloss), ece=r4(getattr(r, "ece5", None))) for r in S.itertuples(index=False) if r.model in NAMES]
D = pd.read_csv(R + "perclass_diffs.csv")
diffs = [dict(protocol=r.protocol, id=r.comparator, name=NAMES[r.comparator], family=FAMILY[r.comparator], metric=r.metric, diff=r4(r.diff), lo=r4(r.lo), hi=r4(r.hi), excl0=bool(r.excl0)) for r in D.itertuples(index=False)]
C = pd.read_csv(R + "conflict_errors.csv")
conflict = [dict(id=r.model, name=NAMES[r.model], cons=r4(100 * r.acc_consistent), cons_lo=r4(100 * r.acc_consistent_lo), cons_hi=r4(100 * r.acc_consistent_hi),
                 conf=r4(100 * r.acc_conflicting), conf_lo=r4(100 * r.acc_conflicting_lo), conf_hi=r4(100 * r.acc_conflicting_hi), share=r4(100 * r.share_errors_on_conflicting))
            for r in C.itertuples(index=False) if r.model != "Majority"]
F = pd.read_csv(R + "record_flags.csv"); P = pd.read_csv(R + "oof_predictions_grouped.csv"); P = P[P.model == "Stack(RF+XGB)"]
hit = P.assign(c=(P.y == P.pred).astype(int)).groupby("row").c.mean()
records = [[int(r.row), int(r.y), int(r.duplicate), int(r.conflicting), round(float(hit.get(r.row, float("nan"))), 2)] for r in F.itertuples(index=False)]
out = dict(audit=json.load(open(R + "audit.json")), meta=json.load(open(R + "conflict_meta.json")), hold=json.load(open(R + "original_split_results.json")), models=models, diffs=diffs, conflict=conflict, records=records,
           order=list(NAMES), classes=["Insomnia", "No recorded disorder", "Sleep apnea"])
open(os.path.join(H, "data.js"), "w").write("window.DATA = " + json.dumps(out, separators=(",", ":")) + ";\n"); print("data.js written:", len(models), "model rows,", len(diffs), "differences,", len(records), "records")
