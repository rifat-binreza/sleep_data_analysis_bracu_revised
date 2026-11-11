"""Generate every table and every in-text number of the manuscript from the result files (no hand-typed results)."""
import json, numpy as np, pandas as pd, os, re
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results", ""); O = os.path.join(HERE, "..", "..", "paper_tables", ""); os.makedirs(O, exist_ok=True)
S = pd.read_csv(R + "summary_all_protocols.csv"); P = pd.read_csv(R + "paired_vs_stack.csv"); A = json.load(open(R + "audit.json")); E = json.load(open(R + "explain_results.json"))
OS = json.load(open(R + "original_split_results.json")); TG = json.load(open(R + "train_gap.json")); NS = pd.read_csv(R + "nested_summary.csv"); NP = pd.read_csv(R + "nested_paired.csv")
FA = pd.read_csv(R + "feature_ablation_folds.csv"); L = pd.read_csv(R + "lime_audit.csv", keep_default_na=False, na_values=[""]); PP = pd.read_csv(R + "perm_importance_block.csv"); VIF = E["vif"]
lab = {"Majority": "Majority class", "Tree(d=3)": "Decision tree (depth 3)", "LogReg": "Logistic regression", "SVM(rbf)": "SVM (RBF)", "MLP": "MLP (32,16)", "FT-Transformer": "FT-Transformer", "RF": "Random forest", "ExtraTrees": "Extra trees",
       "HistGB": "HistGradientBoosting", "LightGBM": "LightGBM", "CatBoost": "CatBoost", "XGB": "XGBoost", "SoftVote(RF+XGB)": "Soft voting (RF+XGB)", "Stack(RF+XGB)": "\\textbf{Stacking (RF+XGB)}"}
order = ["Majority", "Tree(d=3)", "LogReg", "SVM(rbf)", "MLP", "FT-Transformer", "RF", "ExtraTrees", "HistGB", "LightGBM", "CatBoost", "XGB", "SoftVote(RF+XGB)", "Stack(RF+XGB)"]
pc = lambda x, d=1: f"{100 * x:.{d}f}"; f3 = lambda x: f"{x:.3f}"; f2 = lambda x: f"{x:.2f}"
def pval(p): return "$<$0.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
M = {}   # macros
def mac(name, val): M[name] = val
# ---------- audit
for k, v in dict(NRec=A["n_rows"], NUnique=A["unique_rows"], NVec=A["unique_predictor_vectors"], NDup=A["exact_duplicate_rows_excluding_id"], NConflictVec=A["predictor_vectors_with_conflicting_labels"],
                 NConflictRows=A["rows_in_conflicting_vectors"], NNone=A["class_counts"]["None"], NIns=A["class_counts"]["Insomnia"], NApn=A["class_counts"]["Sleep Apnea"],
                 NUniqNone=A["unique_by_class"]["None"], NUniqIns=A["unique_by_class"]["Insomnia"], NUniqApn=A["unique_by_class"]["Sleep Apnea"],
                 OrigLeak=OS["leakage"]["test_rows_with_identical_train_vector"], OrigNovel=OS["leakage"]["test_rows"] - OS["leakage"]["test_rows_with_identical_train_vector"],
                 OrigSameLabel=A["orig_split_test_rows_with_identical_vector_and_same_label"]).items(): mac(k, v)
mac("Ceiling", pc(A["bayes_ceiling_accuracy_on_374"])); mac("CeilingRows", int(round(A["bayes_ceiling_accuracy_on_374"] * A["n_rows"])))
mac("TrainAccRF", pc(TG["RF_train_acc_all374"])); mac("TrainAccStack", pc(TG["Stack_train_acc_all374"])); mac("TrainAccXGB", pc(TG["XGB_train_acc_all374"]))
# ---------- original split
for m, key in [("Stack(RF+XGB)", "Stack"), ("SoftVote(RF+XGB)", "Vote"), ("RF", "RF"), ("XGB", "XGB"), ("LogReg", "LR")]:
    o = OS[m]; mac(f"Orig{key}Correct", o["correct"]); mac(f"Orig{key}Acc", pc(o["test_acc"], 2)); mac(f"Orig{key}CPlo", pc(o["clopper_pearson_95"][0])); mac(f"Orig{key}CPhi", pc(o["clopper_pearson_95"][1]))
    mac(f"Orig{key}F", f2(o["macro_f1"])); mac(f"Orig{key}Train", pc(o["train_acc"])); mac(f"Orig{key}NovelAcc", pc(o["acc_on_novel_test_rows"]))
    mac(f"Orig{key}NovelCorrect", int(round(o["acc_on_novel_test_rows"] * o["n_novel"])))
pv = OS["paired_stack_vs_softvote"]; mac("OrigStackOnly", pv["stack_only_correct"]); mac("OrigVoteOnly", pv["vote_only_correct"]); mac("OrigMcnemar", f"{pv['mcnemar_exact_p']:.2f}")
mac("OrigBootLo", f"{pv['boot_ci_pp'][0]:.1f}"); mac("OrigBootHi", f"{pv['boot_ci_pp'][1]:.1f}"); mac("OrigMDE", f"{OS['power_note']['mde_pp_80power']:.1f}")
# ---------- CV summaries
def g_(p, m, k): return S[(S.protocol == p) & (S.model == m)].iloc[0][k]
for p, pn in [("grouped", "G"), ("dedup", "D"), ("naive", "N")]:
    for m, key in [("Stack(RF+XGB)", "Stack"), ("SoftVote(RF+XGB)", "Vote"), ("LogReg", "LR"), ("RF", "RF"), ("XGB", "XGB"), ("Majority", "Maj"), ("Tree(d=3)", "Tree"), ("SVM(rbf)", "SVM"), ("CatBoost", "Cat"), ("FT-Transformer", "FTT")]:
        if len(S[(S.protocol == p) & (S.model == m)]) == 0: continue
        mac(f"Acc{key}{pn}", pc(g_(p, m, "acc"))); mac(f"F{key}{pn}", f2(g_(p, m, "macro_f1"))); mac(f"Brier{key}{pn}", f3(g_(p, m, "brier"))); mac(f"Auc{key}{pn}", f2(g_(p, m, "auc")))
        if "acc_bs_lo" in S.columns and not np.isnan(g_(p, m, "acc_bs_lo")): mac(f"CIlo{key}{pn}", pc(g_(p, m, "acc_bs_lo"))); mac(f"CIhi{key}{pn}", pc(g_(p, m, "acc_bs_hi")))
        mac(f"FoldSD{key}{pn}", pc(g_(p, m, "acc_sd_fold"))); mac(f"RepSD{key}{pn}", pc(g_(p, m, "acc_sd_rep"), 2)); mac(f"Ece{key}{pn}", f3(g_(p, m, "ece5"))); mac(f"Ll{key}{pn}", f3(g_(p, m, "logloss")))
mac("InflationNaiveGrouped", f"{100 * (g_('naive', 'Stack(RF+XGB)', 'acc') - g_('grouped', 'Stack(RF+XGB)', 'acc')):.1f}")
mac("DropGroupedDedup", f"{100 * (g_('grouped', 'Stack(RF+XGB)', 'acc') - g_('dedup', 'Stack(RF+XGB)', 'acc')):.1f}")
# paired
def pp(p, c, k): return P[(P.protocol == p) & (P.comparator == c)].iloc[0][k]
for p, pn in [("grouped", "G"), ("dedup", "D")]:
    for c, key in [("SoftVote(RF+XGB)", "Vote"), ("LogReg", "LR"), ("RF", "RF"), ("XGB", "XGB"), ("Tree(d=3)", "Tree"), ("FT-Transformer", "FTT")]:
        mac(f"Diff{key}{pn}", f"{pp(p, c, 'mean_acc_diff_pp'):+.1f}"); mac(f"DiffLo{key}{pn}", f"{pp(p, c, 'boot_lo_pp'):+.1f}"); mac(f"DiffHi{key}{pn}", f"{pp(p, c, 'boot_hi_pp'):+.1f}")
        mac(f"McnP{key}{pn}", f2(pp(p, c, "mcnemar_p_median"))); mac(f"NBp{key}{pn}", f2(pp(p, c, "nb_p")))
mac("VoteDiscStack", f"{pp('grouped', 'SoftVote(RF+XGB)', 'discordant_stack_only_mean'):.1f}"); mac("VoteDiscVote", f"{pp('grouped', 'SoftVote(RF+XGB)', 'discordant_comp_only_mean'):.1f}")
mac("FoldsBetterVote", int(pp("grouped", "SoftVote(RF+XGB)", "folds_stack_better"))); mac("FoldsTieVote", int(pp("grouped", "SoftVote(RF+XGB)", "folds_tie"))); mac("FoldsWorseVote", int(pp("grouped", "SoftVote(RF+XGB)", "folds_comp_better")))
sig_g = P[(P.protocol == "grouped") & (P.comparator != "Majority") & (P.boot_lo_pp > 0)].comparator.tolist(); mac("SigGrouped", ", ".join(lab[c].replace("\\textbf{", "").replace("}", "") for c in sig_g) if sig_g else "none")
# nested
for p, pn in [("grouped", "G"), ("dedup", "D")]:
    for m, key in [("Stack(RF+XGB)", "Stack"), ("SoftVote(RF+XGB)", "Vote"), ("LogReg", "LR"), ("RF", "RF"), ("XGB", "XGB")]:
        r = NS[(NS.protocol == p) & (NS.model == m)].iloc[0]; mac(f"NAcc{key}{pn}", pc(r.acc)); mac(f"NF{key}{pn}", f2(r.macro_f1)); mac(f"NCIlo{key}{pn}", pc(r.acc_bs_lo)); mac(f"NCIhi{key}{pn}", pc(r.acc_bs_hi))
    for c, key in [("SoftVote(RF+XGB)", "Vote"), ("LogReg", "LR"), ("RF", "RF"), ("XGB", "XGB")]:
        r = NP[(NP.protocol == p) & (NP.comparator == c)].iloc[0]; mac(f"NDiff{key}{pn}", f"{r.diff_pp:+.1f}"); mac(f"NDiffLo{key}{pn}", f"{r.lo:+.1f}"); mac(f"NDiffHi{key}{pn}", f"{r.hi:+.1f}")
# power: observations needed to detect the single-split difference at 80% power given its discordance rate
pd_ = (pv["stack_only_correct"] + pv["vote_only_correct"]) / OS["power_note"]["n"]; dl = 1 / OS["power_note"]["n"]; mac("NReq", int(round((1.959964 + 0.841621) ** 2 * pd_ / dl ** 2, -1)))
# explanation
mac("ShapRhoStack", f2(E["Stack"]["mean_pairwise_spearman"])); mac("ShapRhoVote", f2(E["SoftVote"]["mean_pairwise_spearman"])); mac("ShapMinRhoStack", f2(E["Stack"]["min_pairwise_spearman"])); mac("ShapMinRhoVote", f2(E["SoftVote"]["min_pairwise_spearman"]))
mac("ShapJacStack", f2(E["Stack"]["mean_top3_jaccard"])); mac("ShapJacVote", f2(E["SoftVote"]["mean_top3_jaccard"])); mac("ShapStackVote", f2(E["stack_vs_vote_spearman_of_mean_importance"]))
for m, key in [("Stack", "Stack"), ("SoftVote", "Vote")]:
    for b, bk in [("Sleep/stress block", "Sleep"), ("Blood-pressure block", "BP"), ("Activity block", "Act")]: mac(f"Block{bk}{key}", f3(E[m]["block_importance_mean"][b]))
    mi = E[m]["mean_importance"]; top = sorted(mi, key=lambda k: -mi[k])[:3]; mac(f"Top{key}", ", ".join(t.replace("_", " ") for t in top))
for k, v in VIF.items(): mac("VIF" + re.sub("[^A-Za-z]", "", k), f"{v:.1f}")
rho = pd.read_csv(R + "spearman_all374.csv", index_col=0)
for a, b, nm in [("Sleep Duration", "Quality of Sleep", "SleepQual"), ("Quality of Sleep", "Stress Level", "QualStress"), ("Systolic_BP", "Diastolic_BP", "BP"), ("Physical Activity Level", "Daily Steps", "ActSteps"), ("Sleep Duration", "Stress Level", "SleepStress"), ("Stress Level", "Heart Rate", "StressHR")]:
    mac("Rho" + nm, f"{rho.loc[a, b]:.2f}")
pm = PP.set_index(["model", "unit"])["mean"]; 
for m, key in [("Stack", "Stack"), ("SoftVote", "Vote")]:
    mac(f"PermBPblock{key}", f3(pm[(m, "[BLOCK] Blood-pressure block")])); mac(f"PermSBP{key}", f3(pm[(m, "Systolic_BP")])); mac(f"PermDBP{key}", f3(pm[(m, "Diastolic_BP")])); mac(f"PermSleepblock{key}", f3(pm[(m, "[BLOCK] Sleep/stress block")]))
mac("LimeRsqStackMin", f2(L[L.model == "Stack"].lime_R2_mean.min())); mac("LimeRsqStackMax", f2(L[L.model == "Stack"].lime_R2_mean.max())); mac("LimeRsqVoteMin", f2(L[L.model == "SoftVote"].lime_R2_mean.min())); mac("LimeRsqVoteMax", f2(L[L.model == "SoftVote"].lime_R2_mean.max()))
mac("LimeRhoStackMin", f2(L[L.model == "Stack"].weight_rank_spearman_over_seeds.min())); mac("LimeRhoVoteMax", f2(L[L.model == "SoftVote"].weight_rank_spearman_over_seeds.max()))
# feature ablation
fa = FA.groupby(["set", "model", "repeat"])["acc"].mean().groupby(["set", "model"]).mean()
for sn, key in [("All predictors", "All"), ("Without sleep duration & quality", "NoSQ"), ("Without all sleep/stress (dur., qual., stress)", "NoSQS")]:
    for m, mk in [("Stack(RF+XGB)", "Stack"), ("SoftVote(RF+XGB)", "Vote"), ("LogReg", "LR"), ("XGB", "XGB"), ("RF", "RF")]: mac(f"Abl{key}{mk}", pc(fa[(sn, m)]))
with open(O + "numbers.tex", "w") as f:
    for k, v in M.items(): f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
# ================= tables =================
def ci(p, m): return f"{pc(g_(p, m, 'acc'))} [{pc(g_(p, m, 'acc_bs_lo'))}, {pc(g_(p, m, 'acc_bs_hi'))}]"
t = ["\\begin{tabular}{@{}lcccccc@{}}", "\\toprule", "Model & Accuracy (\\%) [95\\% CI] & Macro-F1 & Macro AUC & Brier & Log-loss & ECE$_5$\\\\", "\\midrule"]
for m in order: t.append(f"{lab[m]} & {ci('grouped', m)} & {f2(g_('grouped', m, 'macro_f1'))} & {f2(g_('grouped', m, 'auc'))} & {f3(g_('grouped', m, 'brier'))} & {f3(g_('grouped', m, 'logloss'))} & {f3(g_('grouped', m, 'ece5'))}\\\\" + ("\\midrule" if m in ("Majority", "SVM(rbf)", "CatBoost") else ""))
t += ["\\bottomrule", "\\end{tabular}"]; open(O + "tab_main.tex", "w").write("\n".join(t))
t = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule", "Model & Accuracy (\\%) [95\\% CI] & Macro-F1 & Brier & Log-loss\\\\", "\\midrule"]
for m in order: t.append(f"{lab[m]} & {ci('dedup', m)} & {f2(g_('dedup', m, 'macro_f1'))} & {f3(g_('dedup', m, 'brier'))} & {f3(g_('dedup', m, 'logloss'))}\\\\" + ("\\midrule" if m in ("Majority", "SVM(rbf)", "CatBoost") else ""))
t += ["\\bottomrule", "\\end{tabular}"]; open(O + "tab_dedup.tex", "w").write("\n".join(t))
t = ["\\begin{tabular}{@{}lccccc@{}}", "\\toprule", "Comparator & $\\Delta$ accuracy (pp) [95\\% CI] & Folds: stack $>$/=/$<$ & Median McNemar $p$ & Repeats $p<0.05$ & Corrected $t$-test $p$\\\\", "\\midrule"]
for p, name in [("grouped", "Grouped CV (374 records)"), ("dedup", "Unique rows only (132 records)")]:
    t.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{name}}}}}\\\\")
    for m in [o for o in order if o != "Stack(RF+XGB)"]:
        r = P[(P.protocol == p) & (P.comparator == m)].iloc[0]
        t.append(f"{lab[m]} & {r.mean_acc_diff_pp:+.1f} [{r.boot_lo_pp:+.1f}, {r.boot_hi_pp:+.1f}] & {int(r.folds_stack_better)}/{int(r.folds_tie)}/{int(r.folds_comp_better)} & {pval(r.mcnemar_p_median)} & {int(r.repeats_p_lt_05)}/10 & {pval(r.nb_p)}\\\\")
    t.append("\\midrule")
t[-1] = "\\bottomrule"; t.append("\\end{tabular}"); open(O + "tab_paired.tex", "w").write("\n".join(t))
t = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule", "Model & Accuracy (\\%) [95\\% CI] & Macro-F1 & Brier & Log-loss\\\\", "\\midrule"]
for p, name in [("grouped", "Grouped CV (374 records)"), ("dedup", "Unique rows only (132 records)")]:
    t.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{name}}}}}\\\\")
    for m in ["LogReg", "RF", "XGB", "SoftVote(RF+XGB)", "Stack(RF+XGB)"]:
        r = NS[(NS.protocol == p) & (NS.model == m)].iloc[0]; t.append(f"{lab[m]} & {pc(r.acc)} [{pc(r.acc_bs_lo)}, {pc(r.acc_bs_hi)}] & {f2(r.macro_f1)} & {f3(r.brier)} & {f3(r.logloss)}\\\\")
    t.append("\\midrule")
t[-1] = "\\bottomrule"; t.append("\\end{tabular}"); open(O + "tab_nested.tex", "w").write("\n".join(t))
t = ["\\begin{tabular}{@{}lcccccc@{}}", "\\toprule", "Model & Correct/75 & Accuracy (\\%) [exact 95\\% CI] & Macro-F1 & Train acc. (\\%) & Test rows with unseen vector: correct/7\\\\", "\\midrule"]
for m in ["LogReg", "RF", "XGB", "SoftVote(RF+XGB)", "Stack(RF+XGB)"]:
    o = OS[m]; t.append(f"{lab[m]} & {o['correct']} & {pc(o['test_acc'])} [{pc(o['clopper_pearson_95'][0])}, {pc(o['clopper_pearson_95'][1])}] & {f2(o['macro_f1'])} & {pc(o['train_acc'])} & {int(round(o['acc_on_novel_test_rows'] * o['n_novel']))}/{o['n_novel']}\\\\")
t += ["\\bottomrule", "\\end{tabular}"]; open(O + "tab_orig.tex", "w").write("\n".join(t).replace("Correct/75 & Accuracy (\\%) [exact 95\\% CI] & Macro-F1 & Train acc. (\\%) & Test rows with unseen vector: correct/7", "Correct/75 & Accuracy (\\%) [exact 95\\% CI] & Macro-F1 & Train acc.\\ (\\%) & Unseen-vector rows correct"))
short = lambda s: ", ".join(x.split("(")[0].replace("_", " ").replace("Occupation Teacher", "Occ.\\ Teacher") for x in s.split(";")[:3])
t = ["\\begin{tabular}{@{}llllcccl@{}}", "\\toprule", "Model & Person ID & Reference & Predicted & $P$(Ins./None/Apnea) & LIME $R^2$ & Rank $\\rho$ & Top-3 local features\\\\", "\\midrule"]
for _, r in L.iterrows():
    t.append(f"{'Stacking' if r.model == 'Stack' else 'Soft voting'} & {r.person_id} & {r.true_label} & {r.predicted} & {r.p_insomnia:.2f}/{r.p_none:.2f}/{r.p_apnea:.2f} & {r.lime_R2_mean:.2f} & {r.weight_rank_spearman_over_seeds:.2f} & {short(r.top5)}\\\\" + ("\\midrule" if r.model == "SoftVote" and r.test_position != L.test_position.iloc[-1] else ""))
t += ["\\bottomrule", "\\end{tabular}"]; open(O + "tab_lime.tex", "w").write("\n".join(t))
t = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule", "Predictor set & Stacking & Soft voting & Logistic regression\\\\", "\\midrule"]
for sn, key in [("All predictors", "All"), ("Without sleep duration \\& quality", "NoSQ"), ("Without sleep duration, quality \\& stress", "NoSQS")]:
    t.append(f"{sn} & {M['Abl' + key + 'Stack']} & {M['Abl' + key + 'Vote']} & {M['Abl' + key + 'LR']}\\\\")
t += ["\\bottomrule", "\\end{tabular}"]; open(O + "tab_ablation.tex", "w").write("\n".join(t))
mi_s, mi_v = E["Stack"]["mean_importance"], E["SoftVote"]["mean_importance"]; rs, rv = E["Stack"]["rank_range"], E["SoftVote"]["rank_range"]
t = ["\\begin{tabular}{@{}lcc@{}}", "\\toprule", "Quantity & Stacking & Soft voting\\\\", "\\midrule",
     f"Mean pairwise rank correlation across 10 runs (min) & {M['ShapRhoStack']} ({M['ShapMinRhoStack']}) & {M['ShapRhoVote']} ({M['ShapMinRhoVote']})\\\\",
     f"Mean pairwise Jaccard overlap of top-3 sets & {M['ShapJacStack']} & {M['ShapJacVote']}\\\\",
     f"Top-3 predictors (mean $|\\phi|$) & {M['TopStack']} & {M['TopVote']}\\\\",
     f"Blood-pressure block, mean $|\\phi|$ (SBP+DBP) & {M['BlockBPStack']} & {M['BlockBPVote']}\\\\", f"Sleep/stress block (duration+quality+stress) & {M['BlockSleepStack']} & {M['BlockSleepVote']}\\\\",
     f"Activity block (activity+steps) & {M['BlockActStack']} & {M['BlockActVote']}\\\\",
     f"Held-out $\\Delta$log-loss, SBP / DBP alone & {M['PermSBPStack']} / {M['PermDBPStack']} & {M['PermSBPVote']} / {M['PermDBPVote']}\\\\",
     f"Held-out $\\Delta$log-loss, BP block permuted jointly & {M['PermBPblockStack']} & {M['PermBPblockVote']}\\\\", "\\bottomrule", "\\end{tabular}"]
open(O + "tab_shap.tex", "w").write("\n".join(t)); print(len(M), "macros;", sorted(os.listdir(O)))

# keep the optional TabPFN rows (filled by integrate_tabpfn.py through tabpfn_numbers.tex) in the generated tables
for _f, _hooks in [("tab_main.tex", ["\\TabPFNRowMain"]), ("tab_paired.tex", ["\\TabPFNRowPairedG", "\\TabPFNRowPairedD"])]:
    _t = open(O + _f).read().split("\n"); _i = [k for k, l in enumerate(_t) if l.startswith("FT-Transformer")]
    if len(_i) == len(_hooks) and "TabPFNRow" not in "".join(_t):
        for k, h in sorted(zip(_i, _hooks), reverse=True): _t.insert(k + 1, h)
        open(O + _f, "w").write("\n".join(_t))
