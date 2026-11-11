import numpy as np, pandas as pd, json
from sklearn.model_selection import train_test_split
from common import *
X, y = load(); g = group_ids(X)
out = {}
out["n_rows"] = len(X)
out["class_counts"] = {c:int((y==i).sum()) for i,c in enumerate(CLASSES)}
full = X.copy(); full["y"]=y
out["exact_duplicate_rows_excluding_id"] = int(full.duplicated().sum())
out["unique_rows"] = int(len(full.drop_duplicates()))
out["unique_by_class"] = {c:int(full.drop_duplicates()["y"].eq(i).sum()) for i,c in enumerate(CLASSES)}
out["unique_predictor_vectors"] = int(len(np.unique(g)))
# label conflicts: identical predictor vector, different labels
lab = full.assign(g=g).groupby("g")["y"].nunique()
out["predictor_vectors_with_conflicting_labels"] = int((lab>1).sum())
conf_rows = full.assign(g=g)[full.assign(g=g)["g"].isin(lab[lab>1].index)]
out["rows_in_conflicting_vectors"] = int(len(conf_rows))
# best achievable accuracy by any deterministic function of the predictors (majority label per vector)
maj = full.assign(g=g).groupby("g")["y"].agg(lambda s: s.value_counts().iloc[0]).sum()
out["bayes_ceiling_accuracy_on_374"] = float(maj/len(full))
# original protocol: stratified 80:20 seed 42 -> how many test rows have an identical predictor vector in train?
tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=42)
gt = set(g[tr]); out["orig_split_test_rows"]=len(te)
out["orig_split_test_rows_with_identical_train_vector"] = int(sum(g[i] in gt for i in te))
out["orig_split_test_rows_with_identical_vector_and_same_label"] = int(sum(
    any((g[j]==g[i]) and (y[j]==y[i]) for j in tr) for i in te))
print(json.dumps(out, indent=1))
json.dump(out, open(RESDIR + "audit.json","w"), indent=1)
