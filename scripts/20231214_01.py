# %%
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import surprise

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
    experiment_name: str = "20231214_01"
    fold: int = 5
    fold_csv: str = "../data/train_stkgf_5folds_seed0.csv"

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
N_SELECT: int = 20

# %%
train = pd.read_csv(CSVPath.log_train)
test = pd.read_csv(CSVPath.log_test)
label = pd.read_csv(CSVPath.label_train)

# %%
import torch
import pickle
import gc

train = pd.read_csv(CSVPath.log_train)
test = pd.read_csv(CSVPath.log_test)

n_seqs_series = train.groupby("session_id").agg("max")["seq_no"]
train = train.merge(
    pd.DataFrame({
        "session_id": n_seqs_series.index,
        "max_session": n_seqs_series.tolist(),
    })
)

n_seqs_series = test.groupby("session_id").agg("max")["seq_no"]
test = test.merge(
    pd.DataFrame({
        "session_id": n_seqs_series.index,
        "max_session": n_seqs_series.tolist(),
    })
)

fold = pd.read_csv(Config.fold_csv)
for fold_idx in range(Config.fold):
    train_train = train.merge(fold, on=["session_id"], how="left")
    train_train = train_train.loc[train_train["fold"] != fold_idx]
    assert len(train_train) < len(train)

    df = pd.concat([
            train_train[["session_id", "yad_no"]],
            test[["session_id", "yad_no"]]]
    ).reset_index(drop=True)
    df["seq_no_fixed"] = 1
    df = df.rename(
            columns={"session_id": "user", "yad_no": "item", "seq_no_fixed": "rating"}
    )

    reader = surprise.Reader(rating_scale=(0, 1))
    data = surprise.Dataset.load_from_df(
        df, reader
    ).build_full_trainset()
    item_id_to_yad_id = {
        data.to_inner_iid(yad_id):yad_id for yad_id in (train_train["yad_no"].tolist() + test["yad_no"].tolist())
    }

    model = surprise.NMF(random_state=Config.seed, n_factors=240)
    model.fit(data)

    ratings = torch.mm(
        torch.tensor(model.pu, dtype=torch.float16).cuda(),
        torch.tensor(model.qi.transpose(1, 0), dtype=torch.float16).cuda(),
    ).cpu().numpy()
    ratings += model.bu.reshape(-1, 1).astype(np.float16)
    ratings += model.bi.reshape(1, -1).astype(np.float16)

    np.save(f"../features/NMF_nfactors480_float16_{Config.experiment_name}_fold{fold_idx}.npy", ratings)
    pickle.dump(
        item_id_to_yad_id, open(f"../features/NMF_nfactors480_float16_{Config.experiment_name}_fold{fold_idx}_item_id_to_yad_id.pkl", "wb")
    )


    del model, ratings
    gc.collect()

# %%
raise NotImplementedError()

# %%
import torch

"""
ratings = torch.mm(
    torch.tensor(model.pu, dtype=torch.float16).cuda(),
    torch.tensor(model.qi.transpose(1, 0), dtype=torch.float16).cuda(),
).cpu().numpy()
ratings += model.bu.reshape(-1, 1).astype(np.float16)
ratings += model.bi.reshape(1, -1).astype(np.float16)
"""

# %%
#np.save("../features/NMF_nfactors480_float32_v20231213_01.npy", ratings)

# %%


# %%
train = pd.read_csv(CSVPath.log_train)
test = pd.read_csv(CSVPath.log_test)
label = pd.read_csv(CSVPath.label_train)

# %%
def pad_yad_no(yad_idxs_strs):
    pad_yad_idxs = []
    for yad_idxs_str in yad_idxs_strs:
        yad_idxs = yad_idxs_str.split(" ")
        yad_idxs = [int(yad_idx.strip()) for yad_idx in yad_idxs if yad_idx.strip() != ""]
        pad_yad_idx = [-1] * (10 - len(yad_idxs)) + yad_idxs
        pad_yad_idxs.append(pad_yad_idx)
    return pad_yad_idxs

# %%
train["yad_no"] = train["yad_no"].astype(str) + " "
pad_yad_idxs = pad_yad_no(train.groupby("session_id").sum()["yad_no"].tolist())

# %%
df_train_dict = {
    "session_id": train["session_id"].unique().tolist(),
}
for i in range(10):
    df_train_dict[f"seen_{i}"] = np.array(pad_yad_idxs)[:, i]

df_train = pd.DataFrame(df_train_dict)

# %%
df_train = df_train.merge(label, on="session_id", how="left")

# %%
#df_train["fold"] = validate(
#    "StratifiedGroupKFold", df_train, 5,
#    groups=df_train["session_id"].tolist(), y=df_train["yad_no"].tolist(),
#)
#df_train[["session_id", "yad_no", "fold"]].to_csv("../data/train_stkgf_5folds_seed0.csv", index=False)

fold = pd.read_csv(Config.fold_csv)
df_train = df_train.merge(fold, on=["session_id"], how="left")

# %%
df_train.head(3)

# %%
import itertools
import pickle


def get_flatten(l: list):
    return list(itertools.chain.from_iterable(l))

for fold_idx in range(Config.fold):
    ratings = np.load(f"../features/NMF_nfactors480_float16_{Config.experiment_name}_fold{fold_idx}.npy")
    with open(f"../features/NMF_nfactors480_float16_{Config.experiment_name}_fold{fold_idx}_item_id_to_yad_id.pkl", mode="rb") as f:
        item_id_to_yad_id = pickle.load(f)

    top_idxs = []
    for rating in ratings:
        top_idx = get_top_K(rating, N_SELECT)
        top_idxs.append([item_id_to_yad_id[idx] for idx in top_idx.tolist()[::-1]])

    top_idxs = top_idxs[:train["session_id"].nunique()]

    nmf_select = pd.DataFrame({
        "session_id": get_flatten(
            [[idx] * N_SELECT for idx in train["session_id"].unique().tolist()]
        ),
        "yad_no": get_flatten(top_idxs)
    })
    nmf_select["label"] = 0

    pos_sample = label[
        ["session_id", "yad_no"]
    ]
    pos_sample["label"] = 1

    label_sample = pd.concat([nmf_select, pos_sample], axis=0).drop_duplicates(
                        subset=["session_id", "yad_no"], keep="last"
                    ).reset_index(drop=True)
    label_sample.head(5)

    _df_train = df_train.drop(["yad_no"], axis=1)
    _df_train = _df_train.merge(label_sample, on=["session_id"], how="left")
    _df_train = _df_train.groupby(["session_id"]).tail(20).reset_index(drop=True)

    train_folds_v2(
        _df_train,
        [fold_idx],
        Config.seed,
        "ctb", "label",
        # ["session_id", "label", "fold"], ["yad_no"] + [f"seen_{i}" for i in range(10)],
        ["session_id", "label", "fold"], ["yad_no"] + [f"seen_{i}" for i in range(10)],
        f"../models/{Config.experiment_name}_model",
    )

# %% [markdown]
# - yad_no cause leak.

# %%
x_pred_ctb = None
for fold_idx in range(Config.fold):
    ratings = np.load(f"../features/NMF_nfactors480_float16_{Config.experiment_name}_fold{fold_idx}.npy")
    with open(f"../features/NMF_nfactors480_float16_{Config.experiment_name}_fold{fold_idx}_item_id_to_yad_id.pkl", mode="rb") as f:
        item_id_to_yad_id = pickle.load(f)

    top_idxs = []
    for rating in ratings:
        top_idx = get_top_K(rating, N_SELECT)
        top_idxs.append([item_id_to_yad_id[idx] for idx in top_idx.tolist()[::-1]])

    top_idxs = top_idxs[:train["session_id"].nunique()]

    nmf_select = pd.DataFrame({
        "session_id": get_flatten(
            [[idx] * N_SELECT for idx in train["session_id"].unique().tolist()]
        ),
        "yad_no": get_flatten(top_idxs)
    })
    nmf_select["label"] = 0

    pos_sample = label[
        ["session_id", "yad_no"]
    ]
    pos_sample["label"] = 1

    label_sample = pd.concat([nmf_select, pos_sample], axis=0).drop_duplicates(
                        subset=["session_id", "yad_no"], keep="last"
                    ).reset_index(drop=True)
    label_sample.head(5)

    _df_train = df_train.drop(["yad_no"], axis=1)
    _df_train = _df_train.merge(label_sample, on=["session_id"], how="left")
    _df_train = _df_train.groupby(["session_id"]).tail(20).reset_index(drop=True)

    _x_pred_ctb = eval_folds_v2(
        _df_train,
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
"""
x_pred_ctb = eval_folds(
    df_train, Config.fold, Config.seed,
    "ctb", "label",
    ["session_id", "label", "fold"], ["yad_no"] + [f"seen_{i}" for i in range(10)],
    f"../models/{Config.experiment_name}_model"
)
"""

# %%
oof_preds = []
for yad_idxs, scores in zip(
    _df_train["yad_no"].to_numpy().reshape(-1, 20), x_pred_ctb.reshape(-1, 20)
):
    yad_rankings = get_top_K(scores, 10)
    oof_preds.append(yad_idxs[yad_rankings].tolist())

# %%
sample_n = 288698 # 10_000
mapk(
    [[idx] for idx in label["yad_no"].tolist()][:sample_n],
    oof_preds[:sample_n],
)

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
import itertools

def get_flatten(l: list):
    return list(itertools.chain.from_iterable(l))



ratings = np.load("../features/NMF_nfactors480_float16_v20231211_01.npy")
import pickle

with open("../features/NMF_nfactors480_float16_v20231211_01_idxs.pkl", mode="rb") as f:
    item_id_to_yad_id = pickle.load(f)

top_idxs = []
for rating in ratings:
    top_idx = get_top_K(rating, N_SELECT)
    top_idxs.append([item_id_to_yad_id[idx] for idx in top_idx.tolist()[::-1]])

nmf_select = pd.DataFrame({
    "session_id": get_flatten(
        [[idx] * N_SELECT for idx in train["session_id"].unique().tolist()]
        + [[idx] * N_SELECT for idx in test["session_id"].unique().tolist()]
    ),
    "yad_no": get_flatten(top_idxs)
})

test = pd.read_csv(CSVPath.log_test)
test["yad_no"] = test["yad_no"].astype(str) + " "
pad_yad_idxs = pad_yad_no(test.groupby("session_id").sum()["yad_no"].tolist())

df_test_dict = {
    "session_id": test["session_id"].unique().tolist(),
}
for i in range(10):
    df_test_dict[f"seen_{i}"] = np.array(pad_yad_idxs)[:, i]
df_test = pd.DataFrame(df_test_dict)

df_test = df_test.merge(nmf_select, on=["session_id"], how="left")
df_test.head(20)

# %%
X_test = df_test[[col for col in df_test.columns.tolist() if col not in ["session_id", "label", "fold"]]]
preds = predict_catboost(
    X_test, list(range(Config.fold)),
    ["yad_no"] + [f"seen_{i}" for i in range(10)],
    f"../models/{Config.experiment_name}_model",
)

# %%
preds

# %%
y_preds = []
for yad_idxs, scores, last_yado in zip(
    df_test["yad_no"].to_numpy().reshape(-1, 20), preds.reshape(-1, 20), test.groupby("session_id").tail(1)["yad_no"].tolist()
):
    yad_rankings = get_top_K(scores, 11)
    y_preds.append([idx for idx in yad_idxs[yad_rankings].tolist() if idx != last_yado][:10])

# %%
sub = pd.read_csv(CSVPath.submission)

for i in range(10):
    sub[f"predict_{i}"] = np.array(y_preds)[:, i]

# %%
sub.to_csv(f"../sumission_{Config.experiment_name}.csv", index=False)

# %%



