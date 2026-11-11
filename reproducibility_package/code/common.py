"""Shared data loading and model definitions."""
import numpy as np, pandas as pd, hashlib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import os, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(HERE, "..", "results") + os.sep
PAPERDIR = os.path.join(HERE, "..", "paper") + os.sep
CSV = os.environ.get("SLEEP_CSV", os.path.join(HERE, "..", "data", "sleep.csv"))
EXPECTED_SHA256 = "1e83060acd1a84b9a6a7af93f602fc6283ea0ca85369c1db4ed53a8a1bff108b"
CLASSES = ["Insomnia", "None", "Sleep Apnea"]   # 0,1,2  ("None" = literal label in the CSV)
NUM = ["Age","Sleep Duration","Quality of Sleep","Physical Activity Level","Stress Level",
       "Heart Rate","Daily Steps","Systolic_BP","Diastolic_BP"]
CAT = ["Gender","Occupation","BMI Category"]
FEATS = ["Gender","Age","Occupation","Sleep Duration","Quality of Sleep","Physical Activity Level",
         "Stress Level","BMI Category","Heart Rate","Daily Steps","Systolic_BP","Diastolic_BP"]

def load():
    # keep_default_na=False: the literal string "None" must NOT be parsed as missing
    assert hashlib.sha256(open(CSV, "rb").read()).hexdigest() == EXPECTED_SHA256, "unexpected dataset file (SHA-256 mismatch)"
    df = pd.read_csv(CSV, keep_default_na=False, na_values=[""])
    assert df.isna().sum().sum() == 0
    assert set(df["Sleep Disorder"]) == set(CLASSES)
    df = df.drop(columns=["Person ID"])
    df["BMI Category"] = df["BMI Category"].replace({"Normal Weight": "Normal"})
    bp = df["Blood Pressure"].str.split("/", expand=True)
    df["Systolic_BP"] = bp[0].astype(float); df["Diastolic_BP"] = bp[1].astype(float)
    df = df.drop(columns=["Blood Pressure"])
    y = df["Sleep Disorder"].map({c:i for i,c in enumerate(CLASSES)}).values
    X = df[FEATS].reset_index(drop=True)
    return X, y

def group_ids(X):
    """Rows with identical predictor vectors share a group (so duplicates never straddle a split)."""
    key = X.astype(str).agg("|".join, axis=1)
    return pd.factorize(key)[0]

def make_pre(cats=None):
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False, categories=cats)
    return ColumnTransformer([("num", StandardScaler(), NUM), ("cat", oh, CAT)])

def vocab(X):  # categorical vocabulary from a TRAINING partition only
    return [sorted(X[c].unique().tolist()) for c in CAT]

def rf(seed=42):
    return RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5,
                                  min_samples_leaf=1, random_state=seed, n_jobs=1)
def xgb(seed=42, **kw):
    p = dict(n_estimators=800, learning_rate=0.02, max_depth=3, subsample=0.9, colsample_bytree=0.9,
             min_child_weight=3, gamma=0.15, reg_lambda=1.2, objective="multi:softprob",
             tree_method="hist", n_jobs=1, random_state=seed, verbosity=0)
    p.update(kw); return XGBClassifier(**p)

class OOFStack:
    """RF+XGB -> multinomial LR meta-learner with encoded-feature passthrough.
    Algorithm 1: preprocessing is re-fitted inside every inner fold; OOF probabilities and
    fold-transformed passthrough rows form the meta-training matrix; base models and the
    transformer are then refitted on all training data for inference."""
    def __init__(self, C=2.0, n_splits=5, seed=42, rf_fn=rf, xgb_fn=xgb, grouped=True):
        self.C, self.n_splits, self.seed, self.rf_fn, self.xgb_fn, self.grouped = C, n_splits, seed, rf_fn, xgb_fn, grouped
    def fit(self, X, y):
        X = X.reset_index(drop=True); cats = vocab(X); n = len(X)
        g = group_ids(X)
        if self.grouped: skf = StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=self.seed); splits = skf.split(X, y, g)
        else:
            from sklearn.model_selection import StratifiedKFold
            splits = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed).split(X, y)
        k = 3
        P_rf = np.zeros((n,k)); P_x = np.zeros((n,k)); passthru = None
        for tr, va in splits:
            pre = make_pre(cats).fit(X.iloc[tr]); Xt, Xv = pre.transform(X.iloc[tr]), pre.transform(X.iloc[va])
            if passthru is None: passthru = np.zeros((n, Xv.shape[1]))
            P_rf[va] = self.rf_fn(self.seed).fit(Xt, y[tr]).predict_proba(Xv)
            P_x[va]  = self.xgb_fn(self.seed).fit(Xt, y[tr]).predict_proba(Xv)
            passthru[va] = Xv
        Z = np.hstack([passthru, P_rf, P_x])
        self.meta = LogisticRegression(C=self.C, max_iter=5000).fit(Z, y)
        self.pre = make_pre(cats).fit(X); Xa = self.pre.transform(X)
        self.rf_ = self.rf_fn(self.seed).fit(Xa, y); self.xgb_ = self.xgb_fn(self.seed).fit(Xa, y)
        return self
    def predict_proba(self, X):
        Xa = self.pre.transform(X)
        return self.meta.predict_proba(np.hstack([Xa, self.rf_.predict_proba(Xa), self.xgb_.predict_proba(Xa)]))
    def predict(self, X): return self.predict_proba(X).argmax(1)
