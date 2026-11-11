"""TabPFN (v2) on the Sleep Health and Lifestyle Dataset, on exactly the folds used in the manuscript.

Run on Kaggle (see the step-by-step guide).  Needs: Internet ON (to install tabpfn and download the weights),
the dataset "uom190346a/sleep-health-and-lifestyle-dataset" added as Input.  A GPU is optional (CPU takes a few minutes).

Writes to /kaggle/working:
  tabpfn_oof_grouped.csv   out-of-fold predictions, 10 x 5 grouped CV on the 374 records
  tabpfn_oof_dedup.csv     out-of-fold predictions, 10 x 5 CV on the 132 unique rows
  tabpfn_meta.json         package version, device, run time, accuracy summary
The fold assignment of every record is embedded below (copied from split_ids_grouped.csv / split_ids_dedup.csv),
so the folds are identical to those of every other model regardless of library versions.
"""
import os, sys, glob, json, time, hashlib, subprocess
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
import numpy as np, pandas as pd
from sklearn.preprocessing import OrdinalEncoder

DRY = os.environ.get("TABPFN_DRYRUN") == "1"          # developer self-test only: swaps in logistic regression, writes *_DRYRUN files
OUT = os.environ.get("TABPFN_OUT", "/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
FOLDS_GROUPED = [
    "42244220004042224241000041411111101001111112121211113333133133444443113333333333334430224444323232003330303003030300000000000434334334334040404401214332222222222114440332222244434444442230343144444444422201111212222222240313010101010101331331331314314422222222222102024000022233223233333332322323221111240011000022201144144101010114000000300331111111110000004411114444444444",
    "22244021111112223233222232322244004004444441414144331311411411444412003333222222220021114444230202444444443443030300000000000424224224224343434430322113333333333001123334444422240000001141444244444444433041111313333333342010212121212121001001001011023322223333333100041000044433022022222220200202001111140011333300001100100101010110222222022331444441111111113311113333333333",
    "13311141114140002020333310100044341443331114141411443133233233222203000000333333334423331111213333443330304220000000000000000432332332332121212212413334444444444111101224444400002222220034323222222222244202222424444444420313010101010101301301301013321133334444444423223000022211443433333334344343442222140000444411121111111121212111000000244004111114440000002211112222222222",
    "43300130001010004041111123233311221220001111111111332122122122111132440000222222222210443333134343443332324004040400000000000321221221221212121122411224444444444001130441111133320000004420232333333333344214444444444444430212010101010101241241241410233300004444444013144111144433402422222224244242442222410000222244403333333303030333333333222221111111110000003300003333333333",
    "10033233334341114140000020200022041440001112121211222422322322222232441111222222444422001111234444330004041224040400000000000323223223223232323320414334444444444000013111111111113333332220232233333333344124444444444444430212010101010101241241241413233322224444444002043444444400022022222220200202003333101111000033331133133131313113000000211111444441113333333100003313113111",
    "00044021113134441411000024244444331332221111111111332122222222111102334444222222110013220000131414111113133223030300000000000021221221221010101100344443333333333332240114444444424444442223242344444444433211111313333333340212010101010101231231231311244400003333333400030000033300322322222223233232332222223311222200004433433404040443000000211442111112220000004411114444444444",
    "03311410001013333331333312122233011114441110101011332222022022000042441111222222003322441111242222000004041442020200000000000321221221221212121120432004444444444224434113333333313333332220232033333333344004444444444444430212010101010101241241241412210000004444444412144111144433442422222224244242443333020033222244413311311313131331000000344221111111110000003300003333333333",
    "11133330003031111112000004044422331330001112121211003433033033444423441111333333220020113333201414333330304334040400000000000331331331331212121124243332222222222442224224444422200000003332343144444444422402222222222222240313010101010101301301301014304422222222222300004444400011013033333330300303001111241111333311144411411444444441444444044110111110002222222411112242442444",
    "10033142221213331313333321211133013110003330303033001311311311222202111111222222441130443333334444002222222223030300000000000423223223223030303304434444444444444331122440000022210000001130333033333333344213333434444444432010212121212121021021021212031144444444444004030000033322012022222220200202004444432200000044441111111141414111222222200221333331111111114411114444444444",
    "20022304440404441411000042422200321224441110101011334144244244000034002222444444114410221111134343221112120333030300000000000041441441441212121123312113333333333333333113333333340000004422222122222222233241111313333333320414010101010101431431431313441144443333333042423111122233014044444440400404001111444400000011142222222242424222222222411000111110003333330233330020220222",
]
FOLDS_DEDUP = [
    "330421214311432240232422414330034403404133020001402434203002211324101031113342140043114211240211023214401202203133300210404413132423",
    "133100434121222104230341212243443032324212140033334310102024430434012202231004310402234340103412230442414141413212311001430010002113",
    "421310410213333401231420200043114311220044401440044401022222140314032323104102334232441000233302302304123443111311244332032100412121",
    "013411400111432313300243400420332314323034104130414424212024304231220320113443301420432001131002221144432143221242304021120310001342",
    "303004241011310231310324310003020421120321131142212214142040242134024034040434443130111400022332122122302434221431324430300401333441",
    "243122311301312323303024221124430013402404113040213024203010400044433332411021201101404141433330310214440214442322233102114003214202",
    "243024231434310040213123242430022431203114140011204202034332311130313012041043423113213014134040424243122140100332040421320120140234",
    "101021431042330142232141314443113422024322101123214034020203434440023311012004031004134430013120243421340222202120112403104133443330",
    "223231022022001043232303212404411432034403003213344411214134414123330102134314200321110204412120021030102130144121040134334034304242",
    "140312110120332104001334201324032412043311012320234303322234404104212002312300041010344102413140241434123412014124132243404140033223",
]
FINGERPRINT = "0b336fad55287c61b7033c4bea2f42361d04a320d9d282c0db3724ba9c6c6101"
CLASSES = ["Insomnia", "None", "Sleep Apnea"]
NUM = ["Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level", "Stress Level", "Heart Rate", "Daily Steps", "Systolic_BP", "Diastolic_BP"]
CAT = ["Gender", "Occupation", "BMI Category"]
FEATS = ["Gender", "Age", "Occupation", "Sleep Duration", "Quality of Sleep", "Physical Activity Level", "Stress Level", "BMI Category", "Heart Rate", "Daily Steps", "Systolic_BP", "Diastolic_BP"]

# ---------------------------------------------------------------- data (same preparation as code/common.py)
def find_csv():
    if os.environ.get("SLEEP_CSV"): return os.environ["SLEEP_CSV"]
    c = [f for f in glob.glob("/kaggle/input/**/*.csv", recursive=True) if "sleep" in os.path.basename(f).lower()]
    assert c, "Dataset not found: add 'Sleep Health and Lifestyle Dataset' (uom190346a) with the 'Add Input' button."
    return sorted(c)[0]
def load(path):
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])          # the label "None" must stay a string
    df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")             # some file versions store it as an empty cell
    assert len(df) == 374 and set(df["Sleep Disorder"]) == set(CLASSES), "unexpected dataset file"
    df = df.drop(columns=["Person ID"]); df["BMI Category"] = df["BMI Category"].replace({"Normal Weight": "Normal"})
    bp = df["Blood Pressure"].str.split("/", expand=True); df["Systolic_BP"] = bp[0].astype(float); df["Diastolic_BP"] = bp[1].astype(float)
    canon = "\n".join("|".join([str(r[c]) for c in CAT] + ["%.3f" % float(r[c]) for c in NUM] + [r["Sleep Disorder"]]) for _, r in df.iterrows())
    assert hashlib.sha256(canon.encode()).hexdigest() == FINGERPRINT, (
        "This CSV differs from the file used in the manuscript (row order or values), so the embedded folds would not match. "
        "Upload data/sleep.csv from the reproducibility package as a Kaggle dataset and add it as Input instead.")
    y = df["Sleep Disorder"].map({c: i for i, c in enumerate(CLASSES)}).values
    return df[FEATS].reset_index(drop=True), y
csv = find_csv(); X, y = load(csv); print("data:", csv, X.shape, np.bincount(y))
keep = ~X.assign(y=y).duplicated().values; Xd, yd = X[keep].reset_index(drop=True), y[keep]; assert len(Xd) == 132
g = pd.factorize(X.astype(str).agg("|".join, axis=1))[0]
for s in FOLDS_GROUPED:                                                    # no predictor vector may straddle two folds
    assert pd.Series(list(s)).groupby(g).nunique().max() == 1
PROTO = {"grouped": (X, y, FOLDS_GROUPED), "dedup": (Xd, yd, FOLDS_DEDUP)}

# ---------------------------------------------------------------- model
if not DRY:
    try: import tabpfn
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tabpfn"]); import tabpfn
    import torch
    from tabpfn import TabPFNClassifier
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"; VERSION = tabpfn.__version__
    def make(seed):
        """TabPFN with the open version-2 weights (Hollmann et al., Nature 2025), default settings, no tuning."""
        kw = dict(device=DEVICE, random_state=seed, categorical_features_indices=list(range(len(CAT))))
        try:
            from tabpfn.constants import ModelVersion                      # tabpfn >= 2.5: ask explicitly for the v2 weights
            try: return TabPFNClassifier.create_default_for_version(ModelVersion.V2, **kw), "v2 (create_default_for_version)"
            except TypeError:
                m = TabPFNClassifier.create_default_for_version(ModelVersion.V2); m.set_params(**kw); return m, "v2 (create_default_for_version)"
        except ImportError:
            return TabPFNClassifier(**kw), "package default (v2 for tabpfn < 2.5)"   # older package: v2 is the only version
else:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
    DEVICE, VERSION = "cpu", "DRYRUN"
    def make(seed): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), "DRYRUN logistic regression"

def encode(Xtr, Xte):
    """Categoricals -> integer codes fitted on the training fold only; numerical predictors left unscaled (TabPFN handles scaling)."""
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(Xtr[CAT])
    f = lambda D: np.hstack([enc.transform(D[CAT]), D[NUM].values.astype(float)])
    return f(Xtr), f(Xte)

# ---------------------------------------------------------------- run
t0 = time.time(); meta = dict(tabpfn_version=VERSION, device=DEVICE, csv=os.path.basename(csv)); tag = "_DRYRUN" if DRY else ""
for proto, (XX, yy, FOLDS) in PROTO.items():
    rows = []
    for rep, s in enumerate(FOLDS):
        fold = np.array(list(s), dtype=int); assert len(fold) == len(XX)
        for k in range(5):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            A, B = encode(XX.iloc[tr], XX.iloc[te]); clf, how = make(1000 + rep); clf.fit(A, yy[tr]); P = clf.predict_proba(B)
            assert P.shape == (len(te), 3) and list(clf.classes_) == [0, 1, 2]
            for i, r in enumerate(te):
                rows.append(dict(protocol=proto, repeat=rep, fold=k, model="TabPFN" if not DRY else "DRYRUN", row=int(r), y=int(yy[r]), pred=int(P[i].argmax()),
                                 p_insomnia=float(P[i, 0]), p_none=float(P[i, 1]), p_apnea=float(P[i, 2])))
        print(f"{proto}: repeat {rep + 1}/10 done ({time.time() - t0:.0f}s)", flush=True)
    d = pd.DataFrame(rows); assert len(d) == 10 * len(XX) and d.groupby("repeat").row.nunique().eq(len(XX)).all()
    d.to_csv(os.path.join(OUT, f"tabpfn_oof_{proto}{tag}.csv"), index=False)
    meta[f"accuracy_{proto}"] = float((d.y == d.pred).groupby([d.repeat, d.fold]).mean().mean()); meta["weights"] = how
meta["seconds"] = round(time.time() - t0, 1); json.dump(meta, open(os.path.join(OUT, f"tabpfn_meta{tag}.json"), "w"), indent=1)
print("\nFINISHED\n", json.dumps(meta, indent=1)); print("Download tabpfn_oof_grouped.csv, tabpfn_oof_dedup.csv and tabpfn_meta.json from the Output panel.")
