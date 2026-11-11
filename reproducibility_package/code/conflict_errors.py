"""Error anatomy: where do the grouped-CV errors fall?
Splits the 374 records into those whose predictor vector carries a single outcome label ("consistent")
and those whose vector carries conflicting labels, and reports accuracy on each part, the share of all
errors that fall on conflicting vectors, and the mean reference->prediction flow of the stack.
Reads only released files (no model fitting).  usage: python conflict_errors.py
"""
import os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
CSV = os.environ.get("SLEEP_CSV", os.path.join(HERE, "..", "data", "sleep.csv"))
CLASSES = ["Insomnia", "None", "Sleep Apnea"]
d = pd.read_csv(CSV, keep_default_na=False, na_values=[""])
d["BMI Category"] = d["BMI Category"].replace({"Normal Weight": "Normal"})
key = d.drop(columns=["Person ID", "Sleep Disorder"]).astype(str).agg("|".join, axis=1)
g = pd.factorize(key)[0]
conf = (d.groupby(g)["Sleep Disorder"].transform("nunique") > 1).values            # row lies in a conflicting vector
maj = d.groupby(g)["Sleep Disorder"].transform(lambda s: s.value_counts().index[0]).values
minority = (d["Sleep Disorder"].values != maj)                                        # row carries the minority label of its vector
p = pd.read_csv(os.path.join(RES, "oof_predictions_grouped.csv"))
if os.path.exists(os.path.join(RES, "tabpfn_oof_grouped.csv")): p = pd.concat([p, pd.read_csv(os.path.join(RES, "tabpfn_oof_grouped.csv"))], ignore_index=True)
rng = np.random.default_rng(20261110); B = 4000; ug = np.unique(g); rows = []
for m, q in p.groupby("model"):
    c = (q.assign(c=(q.y == q.pred).astype(float)).pivot(index="row", columns="repeat", values="c").sort_index().mean(axis=1).values)  # per-record accuracy over repeats
    def stats(c, conf):
        e = 1 - c
        return np.array([c[~conf].mean(), c[conf].mean(), e[conf].sum() / e.sum(), c.mean()])
    pt = stats(c, conf); bs = []
    idx_by_g = [np.where(g == k)[0] for k in ug]
    for _ in range(B):                                                               # cluster bootstrap over predictor vectors
        ii = np.concatenate([idx_by_g[k] for k in rng.integers(0, len(ug), len(ug))])
        if conf[ii].all() or (~conf[ii]).all(): continue
        bs.append(stats(c[ii], conf[ii]))
    lo, hi = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
    rows.append(dict(model=m, acc_consistent=pt[0], acc_consistent_lo=lo[0], acc_consistent_hi=hi[0],
                     acc_conflicting=pt[1], acc_conflicting_lo=lo[1], acc_conflicting_hi=hi[1],
                     share_errors_on_conflicting=pt[2], share_lo=lo[2], share_hi=hi[2], acc_all=pt[3],
                     share_errors_on_minority_label=((1 - c)[minority].sum() / (1 - c).sum())))
out = pd.DataFrame(rows); out.to_csv(os.path.join(RES, "conflict_errors.csv"), index=False)
# mean reference -> prediction flow of the stack per repeat, split by conflict status
s = p[p.model == "Stack(RF+XGB)"].copy(); s["conf"] = conf[s.row.values]
flow = (s.groupby(["y", "pred", "conf"]).size() / s.repeat.nunique()).rename("mean_records").reset_index()
flow.to_csv(os.path.join(RES, "stack_flow_grouped.csv"), index=False)
meta = dict(n_conflicting_rows=int(conf.sum()), n_consistent_rows=int((~conf).sum()), n_minority_label_rows=int(minority.sum()),
            ceiling_on_conflicting=float(1 - minority.sum() / conf.sum()))
json.dump(meta, open(os.path.join(RES, "conflict_meta.json"), "w"), indent=1)
# per-record flags used by the overview figure (panel a)
ymap = {"Insomnia": 0, "None": 1, "Sleep Apnea": 2}
pd.DataFrame(dict(row=np.arange(len(d)), y=d["Sleep Disorder"].map(ymap).values, duplicate=d.drop(columns=["Person ID"]).duplicated().values.astype(int),
                  conflicting=conf.astype(int))).to_csv(os.path.join(RES, "record_flags.csv"), index=False)
print(meta); print(out.round(3).to_string(index=False))
# LaTeX macros used by the manuscript
st = out.set_index("model"); comp = st.drop(index="Majority"); s_ = st.loc["Stack(RF+XGB)"]
fl = flow[(flow.pred == 1) & (flow.y != 1)]
mac = {"ConfRows": meta["n_conflicting_rows"], "ConsRows": meta["n_consistent_rows"], "ConfMinority": meta["n_minority_label_rows"],
       "ConfCeil": f"{100*meta['ceiling_on_conflicting']:.1f}", "ConfRowShare": f"{100*meta['n_conflicting_rows']/374:.0f}",
       "AccConsStack": f"{100*s_.acc_consistent:.1f}", "AccConsStackLo": f"{100*s_.acc_consistent_lo:.1f}", "AccConsStackHi": f"{100*s_.acc_consistent_hi:.1f}",
       "AccConfStack": f"{100*s_.acc_conflicting:.1f}", "AccConfStackLo": f"{100*s_.acc_conflicting_lo:.1f}", "AccConfStackHi": f"{100*s_.acc_conflicting_hi:.1f}",
       "ConfShareStack": f"{100*s_.share_errors_on_conflicting:.0f}", "ConfShareStackLo": f"{100*s_.share_lo:.0f}", "ConfShareStackHi": f"{100*s_.share_hi:.0f}",
       "AccConsMin": f"{100*comp.acc_consistent.min():.1f}", "AccConsMax": f"{100*comp.acc_consistent.max():.1f}",
       "AccConfMin": f"{100*comp.acc_conflicting.min():.1f}", "AccConfMax": f"{100*comp.acc_conflicting.max():.1f}",
       "ConfShareMin": f"{100*comp.share_errors_on_conflicting.min():.0f}", "ConfShareMax": f"{100*comp.share_errors_on_conflicting.max():.0f}",
       "FlowToNoneAll": f"{fl.mean_records.sum():.1f}", "FlowToNoneConf": f"{fl[fl.conf].mean_records.sum():.1f}"}
tex = "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items())
for dst in [os.path.join(RES, "conflict_numbers.tex"), os.path.join(HERE, "..", "..", "paper_tables", "conflict_numbers.tex")]:
    if os.path.isdir(os.path.dirname(dst)): open(dst, "w").write(tex)
