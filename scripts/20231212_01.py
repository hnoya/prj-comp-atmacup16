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
from src.models import train_folds, eval_folds
from src.metrics import mapk
from src.features import get_base_ranking_df

# %%
@dataclass
class Config:
    data_path: str = "../data/"
    seed: int = 0
    experiment_name: str = "20231212_01"
    fold: int = 5
    fold_csv: str = "../data/train_stkgf_5folds_seed0.csv"

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
train = pd.read_csv(CSVPath.log_train)
test = pd.read_csv(CSVPath.log_test)
label = pd.read_csv(CSVPath.label_train)

# %%
yado = pd.read_csv(CSVPath.yado)

for col in yado.columns.tolist():
    if col != "total_room_cnt":
        yado[col] = yado[col].fillna(0)

yado.head(3)

# %%
train = train.merge(yado, on="yad_no", how="left")
test = test.merge(yado, on="yad_no", how="left")

# %%
df_train, df_test = get_base_ranking_df(train, test)

# %%
df_train = df_train.merge(label, on="session_id", how="left")

# %%
#df_train["fold"] = validate(
#    "StratifiedGroupKFold", df_train, 5,
#    groups=df_train["session_id"].tolist(), y=df_train["yad_no"].tolist(),
#)
#df_train[["session_id", "yad_no", "fold"]].to_csv("../data/train_stkgf_5folds_seed0.csv", index=False)

fold = pd.read_csv(Config.fold_csv)
df_train = df_train.merge(fold, on=["session_id", "yad_no"], how="left")

# %%
df_train.head(3)

# %%
import itertools

def get_flatten(l: list):
    return list(itertools.chain.from_iterable(l))

N_SELECT: int = 20

ratings = np.load("../features/NMF_nfactors480_float16_v20231211_01.npy")
top_idxs = []
for rating in ratings:
    top_idx = get_top_K(rating, N_SELECT)
    top_idxs.append(top_idx.tolist())

nmf_select = pd.DataFrame({
    "session_id": get_flatten(
        [[idx] * N_SELECT for idx in train["session_id"].unique().tolist()]
        + [[idx] * N_SELECT for idx in test["session_id"].unique().tolist()]
    ),
    "yad_no": get_flatten(top_idxs)
})
nmf_select["label"] = 0

pos_sample = label[["session_id", "yad_no"]]
pos_sample["label"] = 1

label_sample = pd.concat([nmf_select, pos_sample], axis=0).drop_duplicates(
                    subset=["session_id", "yad_no"], keep="last"
                ).reset_index(drop=True)
label_sample.head(5)

# %%
df_train = df_train.drop(["yad_no"], axis=1)
df_train = df_train.merge(label_sample, on=["session_id"], how="left")
df_train.head(10)

# %%
df_train["label"].mean()

# %%
# got oom
train_folds(
    df_train, Config.fold, Config.seed,
    "ctb", "label", ["session_id", "label", "fold"], ["yad_no"],
    f"../models/{Config.experiment_name}_model",
)

# train_folds(df_train, 5, SEED, "lgb", "../models/20231211_01_model")

# %%



