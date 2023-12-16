# %%
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import surprise
import torch
import pickle
import gc
import torch
import itertools
import pickle


# %%
import sys
sys.path.append("../")

from src.utils import seed_everything, get_top_K
from src.validate import validate
from src.memory import reduce_mem_usage
from src.models import train_folds, eval_folds, predict_catboost, train_folds_v2, eval_folds_v2
from src.metrics import mapk
from src.features import get_base_ranking_df

# %%
@dataclass
class Config:
    data_path: str = "../data/"
    seed: int = 0
    experiment_name: str = "20231216_05"
    fold: int = 5
    fold_csv: str = "../data/train_stkgf_5folds_seed0.csv"
    n_select: int = 11
    sub_max_k: int = 10
    conditions_csv: str = "../data/nfm_preds_20231216_04.csv"

os.makedirs(os.path.join("../models", Config.experiment_name + "_model"), exist_ok=True)

@dataclass
class CSVPath:
    submission: str = os.path.join(Config.data_path, "sample_submission.csv")
    log_train: str = os.path.join(Config.data_path, "train_log.csv")
    log_test: str = os.path.join(Config.data_path, "test_log.csv")
    label_train: str = os.path.join(Config.data_path, "train_label.csv")
    session_test: str = os.path.join(Config.data_path, "test_session.csv")
    yado: str = os.path.join(Config.data_path, "yado.csv")


seed_everything(Config.seed)

# %%
import torch
import pickle
import gc

# %%
def pad_yad_no(yad_idxs_strs):
    pad_yad_idxs = []
    for yad_idxs_str in yad_idxs_strs:
        yad_idxs = yad_idxs_str.split(" ")
        yad_idxs = [int(yad_idx.strip()) for yad_idx in yad_idxs if yad_idx.strip() != ""]
        pad_yad_idx = [-1] * (10 - len(yad_idxs)) + yad_idxs
        pad_yad_idxs.append(pad_yad_idx)
    return pad_yad_idxs

def get_flatten(l: list):
    return list(itertools.chain.from_iterable(l))

# %%
train = pd.read_csv(CSVPath.log_train)
test = pd.read_csv(CSVPath.log_test)
label = pd.read_csv(CSVPath.label_train)

train["yad_no"] = train["yad_no"].astype(str) + " "
pad_yad_idxs = pad_yad_no(train.groupby("session_id").sum()["yad_no"].tolist())

df_train_dict = {
    "session_id": train["session_id"].unique().tolist(),
}
for i in range(10):
    df_train_dict[f"seen_{i}"] = np.array(pad_yad_idxs)[:, i].tolist()

df_train = pd.DataFrame(df_train_dict)
df_train = df_train.merge(label, on="session_id", how="left")
fold = pd.read_csv(Config.fold_csv)
df_train = df_train.merge(fold, on=["session_id"], how="left")


test["yad_no"] = test["yad_no"].astype(str) + " "
pad_yad_idxs = pad_yad_no(test.groupby("session_id").sum()["yad_no"].tolist())
df_test_dict = {
    "session_id": test["session_id"].unique().tolist(),
}
for i in range(10):
    df_test_dict[f"seen_{i}"] = np.array(pad_yad_idxs)[:, i]
df_test = pd.DataFrame(df_test_dict)

# %%
nmf_conds = pd.read_csv(Config.conditions_csv)

nmf_conds_df = pd.DataFrame({
    "session_id": get_flatten([[idx] * Config.n_select for idx in nmf_conds["session_id"]]),
    "yad_no": get_flatten(list(x.split(" ")[:Config.n_select] for x in nmf_conds["oof_preds"].tolist()))
})
nmf_conds_df["label"] = 0
del nmf_conds; gc.collect()

label = pd.read_csv(CSVPath.label_train)
pos_sample = label[
    ["session_id", "yad_no"]
]
pos_sample["label"] = 1

label_sample = pd.concat([nmf_conds_df, pos_sample], axis=0).drop_duplicates(
    subset=["session_id", "yad_no"], keep="last"
).reset_index(drop=True)
label_sample = label_sample.groupby(["session_id"]).tail(Config.n_select).reset_index(drop=True)

# %%
label_sample["label"].describe()

# %%
df_train = df_train.drop(["yad_no"], axis=1)
df_train = df_train.merge(label_sample, on=["session_id"], how="left")

df_test = df_test.merge(label_sample, on=["session_id"], how="left")

# %%
df_train.head(5)

# %%
df_test.head(5)

# %%
df_train["label"].describe()

# %%
for fold_idx in range(Config.fold):
    train_folds_v2(
        df_train,
        [fold_idx],
        Config.seed,
        "ctb", "label",
        # ["session_id", "label", "fold"], ["yad_no"] + [f"seen_{i}" for i in range(10)],
        ["session_id", "label", "fold"], ["yad_no"] + [f"seen_{i}" for i in range(10)],
        f"../models/{Config.experiment_name}_model",
    )

# %%
x_pred_ctb = None
for fold_idx in range(Config.fold):
    _x_pred_ctb = eval_folds_v2(
        df_train,
        [fold_idx],
        Config.seed,
        "ctb", "label",
        # ["session_id", "label", "fold"], ["yad_no"] + [f"seen_{i}" for i in range(10)],
        ["session_id", "label", "fold"], ["yad_no"] + [f"seen_{i}" for i in range(10)],
        f"../models/{Config.experiment_name}_model",
    )
    if x_pred_ctb is None:
        x_pred_ctb = _x_pred_ctb
    else:
        x_pred_ctb += _x_pred_ctb

# %%
oof_preds = []
for yad_idxs, scores in zip(
    df_train["yad_no"].to_numpy().reshape(-1, Config.n_select), x_pred_ctb.reshape(-1, Config.n_select)
):
    yad_rankings = get_top_K(scores, Config.n_select)
    oof_preds.append(yad_idxs[yad_rankings].tolist())

# %%
sample_n = 288698 # 10_000
mapk(
    [[idx] for idx in label["yad_no"].tolist()][:sample_n],
    oof_preds[:sample_n],
)

# %%


# %%
import pickle
import catboost

def predict_catboost(
    X_test: pd.DataFrame,
    folds: list[int],
    categorical_features: list[str],
    model_path: str = "../models",
):
    y_pred = np.zeros((X_test.shape[0]), dtype="float32")
    for fold in folds:
        model = pickle.load(
            open(os.path.join(model_path, "ctb_fold{}.ctbmodel".format(fold)), "rb")
        )
        y_pred += model.predict(
            catboost.Pool(X_test, cat_features=categorical_features)
        ) / len(folds)
    return y_pred

# %%
X_test = df_test[[col for col in df_test.columns.tolist() if col not in ["session_id", "label", "fold"]]]
preds = predict_catboost(
    X_test,
    list(range(Config.fold)),
    ["yad_no"] + [f"seen_{i}" for i in range(10)],
    f"../models/{Config.experiment_name}_model",
)


# %%
y_preds = []
for yad_idxs, scores, last_yado in zip(
    df_test["yad_no"].to_numpy().reshape(-1, Config.n_select),
    preds.reshape(-1, Config.n_select),
    test.groupby("session_id").tail(1)["yad_no"].tolist()
):
    yad_rankings = get_top_K(scores, 11)
    y_preds.append([idx for idx in yad_idxs[yad_rankings].tolist() if idx != last_yado.strip()][:10])

sub = pd.read_csv(CSVPath.submission)

for i in range(10):
    sub[f"predict_{i}"] = np.array(y_preds)[:, i]

# %%
yad_idxs, scores, last_yado.strip()

# %%
sub.to_csv(f"../sumission_{Config.experiment_name}.csv", index=False)

# %%



