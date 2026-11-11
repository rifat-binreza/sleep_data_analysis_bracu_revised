import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os; _H=os.path.dirname(os.path.abspath(__file__)); R=os.path.join(_H,"..","results","")+""; O=os.path.join(_H,"..","..","figs","")
plt.rcParams.update({"font.family":"serif","font.serif":["STIXGeneral","DejaVu Serif"],"mathtext.fontset":"stix","font.size":9,
 "axes.labelsize":9,"axes.titlesize":9.5,"xtick.labelsize":8.5,"ytick.labelsize":8.5,"legend.fontsize":8,"axes.linewidth":.8,
 "axes.spines.top":False,"axes.spines.right":False,"xtick.major.width":.8,"ytick.major.width":.8,"pdf.fonttype":42,"axes.axisbelow":True})
# Okabe-Ito
BL,OR,GN,VM,SK,GY,RD="#0072B2","#E69F00","#009E73","#CC79A7","#56B4E9","#7f7f7f","#D55E00"
S=pd.read_csv(R+"summary_all_protocols.csv"); A=json.load(open(R+"audit.json")); E=json.load(open(R+"explain_results.json"))
def tag(ax,s): ax.text(-0.02,1.05,s,transform=ax.transAxes,fontweight="bold",fontsize=11,ha="right",va="bottom")
def grid(ax,axis="x"): ax.grid(axis=axis,color="#e3e3e3",lw=.7)

# ---- Fig audit
fig,axs=plt.subplots(1,2,figsize=(7.0,3.0),gridspec_kw=dict(width_ratios=[1,1.3],wspace=.32))
a=axs[0]; v=[A["n_rows"],A["unique_rows"],A["unique_predictor_vectors"]]; lab=["Records","Unique\nrows","Predictor\nvectors"]
bars=a.bar(range(3),v,color=[GY,BL,OR],width=.6,edgecolor="white")
for i,x in enumerate(v): a.text(i,x+8,str(x),ha="center",fontsize=9,fontweight="bold")
a.set_xticks(range(3)); a.set_xticklabels(lab); a.set_ylabel("Count"); a.set_ylim(0,450); grid(a,"y"); tag(a,"(a)")
a.text(0.98,0.97,f"{A['predictor_vectors_with_conflicting_labels']} predictor vectors carry\nconflicting labels: accuracy\nceiling {100*A['bayes_ceiling_accuracy_on_374']:.1f}% on 374 records",transform=a.transAxes,ha="right",va="top",fontsize=7.8,color=RD,bbox=dict(boxstyle="round,pad=0.3",fc="#fdf1ea",ec=RD,lw=.6))
a.set_ylim(0,520)
b=axs[1]; models=["Majority","LogReg","Stack(RF+XGB)"]; w=.26
pro=[("naive","Plain stratified CV",GY),("grouped","Grouped CV",BL),("dedup","Unique rows only",OR)]
for j,(p,l,c) in enumerate(pro):
    d=S[S.protocol==p].set_index("model").loc[models]
    b.bar(np.arange(3)+(j-1)*w,100*d["acc"],w,yerr=[100*(d["acc"]-d["acc_bs_lo"]),100*(d["acc_bs_hi"]-d["acc"])],color=c,label=l,capsize=2.5,error_kw=dict(lw=.9,ecolor="#333"),edgecolor="white")
b.set_xticks(range(3)); b.set_xticklabels(["Majority class","Logistic regression","RF–XGB stack"]); b.set_ylim(40,112); b.set_yticks([40,50,60,70,80,90,100]); b.set_ylabel("Accuracy (%)"); grid(b,"y"); tag(b,"(b)")
b.legend(frameon=False,loc="upper center",ncol=3,columnspacing=1.0,handlelength=1.0,fontsize=7.3,borderaxespad=.1)
plt.savefig(O+"fig2_audit.pdf",bbox_inches="tight"); plt.close()

# ---- Fig models
order=["Majority","Tree(d=3)","LogReg","SVM(rbf)","MLP","FT-Transformer","RF","ExtraTrees","HistGB","LightGBM","CatBoost","XGB","SoftVote(RF+XGB)","Stack(RF+XGB)"]
lab={"FT-Transformer":"FT-Transformer","Tree(d=3)":"Decision tree (depth 3)","LogReg":"Logistic regression","SVM(rbf)":"SVM (RBF)","MLP":"MLP (32,16)","RF":"Random forest","ExtraTrees":"Extra trees","HistGB":"HistGradientBoosting","LightGBM":"LightGBM","CatBoost":"CatBoost","XGB":"XGBoost","SoftVote(RF+XGB)":"Soft voting (RF+XGB)","Stack(RF+XGB)":"Stacking (RF+XGB)","Majority":"Majority class"}
fig,axs=plt.subplots(1,2,figsize=(7.0,4.2),sharey=True,gridspec_kw=dict(wspace=.08))
for ax,p,t,s in zip(axs,["grouped","dedup"],["Grouped CV, 374 records","Unique rows only, 132 records"],["(a)","(b)"]):
    d=S[S.protocol==p].set_index("model").loc[order]; y=np.arange(len(order))[::-1]
    ax.axhspan(-.5,0.5,color="#fbe9e0") 
    ax.axvline(100*d.loc["Majority","acc"],color=GY,ls=":",lw=1)
    ax.errorbar(100*d["acc"],y,xerr=[100*(d["acc"]-d["acc_bs_lo"]),100*(d["acc_bs_hi"]-d["acc"])],fmt="none",ecolor="#555",elinewidth=1.1,capsize=2.5,zorder=2)
    cols=[RD if m.startswith("Stack") else (GY if m=="Majority" else BL) for m in order]
    ax.scatter(100*d["acc"],y,c=cols,s=34,zorder=3,edgecolor="white",linewidth=.6)
    ax.set_yticks(y); ax.set_yticklabels([lab[m] for m in order]); ax.set_xlabel("Accuracy (%), mean and 95% CI"); grid(ax,"x"); ax.set_title(t,fontsize=9,loc="left"); tag(ax,s)
    ax.set_ylim(-.7,len(order)-.3)
    for tl in ax.get_yticklabels():
        if tl.get_text().startswith("Stacking"): tl.set_fontweight("bold")
axs[0].set_xlim(45,97); axs[1].set_xlim(40,92)
plt.savefig(O+"fig3_models.pdf",bbox_inches="tight"); plt.close()

# ---- Fig calibration
fig,axs=plt.subplots(1,2,figsize=(7.0,3.1),gridspec_kw=dict(width_ratios=[1.1,1],wspace=.55))
pr=pd.read_csv(R+"oof_predictions_grouped.csv")
cols={"Stack(RF+XGB)":(RD,"Stacking"),"SoftVote(RF+XGB)":(OR,"Soft voting"),"XGB":(GN,"XGBoost"),"LogReg":(VM,"Logistic reg."),"RF":(GY,"Random forest"),"FT-Transformer":("#8c6d31","FT-Transformer")}
a=axs[0]; a.plot([.3,1],[.3,1],color="#333",ls="--",lw=.8)
for m,(c,l) in cols.items():
    p=pr[pr.model==m]; P_=p[["p_insomnia","p_none","p_apnea"]].values; conf=P_.max(1); hit=(P_.argmax(1)==p["y"].values)
    edges=np.array([0,.5,.7,.85,.95,1.0001]); xs=[];ys=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mk=(conf>=lo)&(conf<hi)
        if mk.sum()>=20: xs.append(conf[mk].mean()); ys.append(hit[mk].mean())
    a.plot(xs,ys,"o-",color=c,label=l,ms=4.5,lw=1.6 if m.startswith("Stack") else 1.1,zorder=3 if m.startswith("Stack") else 2,markeredgecolor="white",markeredgewidth=.5)
a.set_xlabel("Mean predicted top-class probability"); a.set_ylabel("Observed accuracy"); a.set_xlim(.3,1.02); a.set_ylim(.3,1.02); grid(a,"both"); a.legend(frameon=False,loc="lower right",fontsize=7.5); tag(a,"(a)")
sg=S[S.protocol=="grouped"].set_index("model").loc[["SVM(rbf)","CatBoost","SoftVote(RF+XGB)","XGB","Stack(RF+XGB)","LogReg","RF"]]
nm={"SVM(rbf)":"SVM (RBF)","CatBoost":"CatBoost","SoftVote(RF+XGB)":"Soft voting","XGB":"XGBoost","Stack(RF+XGB)":"Stacking","LogReg":"Logistic reg.","RF":"Random forest"}
y=np.arange(len(sg))[::-1]; c=[RD if i.startswith("Stack") else BL for i in sg.index]
b=axs[1]; b.barh(y,sg["brier"],color=c,height=.62,edgecolor="white"); b.errorbar(sg["brier"],y,xerr=[sg["brier"]-sg["brier_lo"],sg["brier_hi"]-sg["brier"]],fmt="none",ecolor="#222",capsize=2.5,elinewidth=1)
b.set_yticks(y); b.set_yticklabels([nm[i] for i in sg.index]); b.set_xlim(.15,.21); b.set_xlabel("Multiclass Brier score (lower is better)"); grid(b,"x"); tag(b,"(b)")
plt.savefig(O+"fig5_calibration.pdf",bbox_inches="tight"); plt.close()

# ---- Fig shap
runs=pd.read_csv(R+"shap_stability_runs.csv"); feats=[c for c in runs.columns if c not in("run","model")]
fig,axs=plt.subplots(1,2,figsize=(7.0,3.5),gridspec_kw=dict(wspace=.62))
for ax,m,t,s in zip(axs,["Stack","SoftVote"],["Stacking","Soft voting"],["(a)","(b)"]):
    M=runs[runs.model==m][feats]; mu=M.mean(); o=mu.sort_values().index.tolist(); y=np.arange(len(o))
    isbp=[f in("Systolic_BP","Diastolic_BP") for f in o]
    ax.barh(y,mu[o],color=[RD if b_ else BL for b_ in isbp],height=.68,edgecolor="white")
    ax.hlines(y,M.min()[o],M.max()[o],color="#222",lw=1.3)
    ax.set_yticks(y); ax.set_yticklabels([f.replace("_"," ").replace("BP","BP") for f in o],fontsize=8); ax.set_xlabel("Mean |SHAP| (bar); range over 10 runs (line)"); grid(ax,"x")
    ax.set_title(f"{t}  ($\\bar\\rho$ = {E[m]['mean_pairwise_spearman']:.2f})",fontsize=9,loc="left"); tag(ax,s)
fig.text(.5,-.02,"Red bars: systolic and diastolic blood pressure",ha="center",fontsize=7.8,color=RD)
plt.savefig(O+"fig6_shap.pdf",bbox_inches="tight"); plt.close()
print("ok")
