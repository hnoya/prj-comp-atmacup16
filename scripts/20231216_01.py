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
    experiment_name: str = "20231216_01"
    fold: int = 5
    fold_csv: str = "../data/train_stkgf_5folds_seed0.csv"
    n_select: int = 20
    sub_max_k: int = 10

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
yado = pd.read_csv(CSVPath.yado)

for col in yado.columns.tolist():
    if col != "total_room_cnt":
        yado[col] = yado[col].fillna(0)

yado.head(3)

# %%
train = pd.read_csv(CSVPath.log_train)
label = pd.read_csv(CSVPath.label_train)

def fix_seq(x):
    if x.session_id != x._session_id:
        return 9
    else:
        return 9 - x._seq_no

n_seqs_series = train.groupby("session_id").agg("max")["seq_no"]
train = train.merge(
    pd.DataFrame({
        "session_id": n_seqs_series.index,
        "max_session": n_seqs_series.tolist(),
    })
)

train["seq_no_fixed"] = 9 - train["max_session"] + train["seq_no"]
label["seq_no_fixed"] = 10
train_with_label = pd.concat([train, label], axis=0).reset_index(
    drop=True
).sort_values(by="session_id", ignore_index=True)

train_with_label.head(3)

# %%
test = pd.read_csv(CSVPath.log_test)

n_seqs_series = test.groupby("session_id").agg("max")["seq_no"]
test = test.merge(
    pd.DataFrame({
        "session_id": n_seqs_series.index,
        "max_session": n_seqs_series.tolist(),
    })
)

test["seq_no_fixed"] = 9 - test["max_session"] + test["seq_no"]

test.head(3)

# %%
df = pd.concat([
        train_with_label[["session_id", "yad_no", "seq_no_fixed"]],
        test[["session_id", "yad_no", "seq_no_fixed"]]]
).reset_index(drop=True)
df["seq_no_fixed"] += 1
df = df.rename(
        columns={"session_id": "user", "yad_no": "item", "seq_no_fixed": "rating"}
)

# %%
df["rating"].describe()

# %%
reader = surprise.Reader(rating_scale=(1, 11))
data = surprise.Dataset.load_from_df(
    df, reader
).build_full_trainset()
item_id_to_yad_id = {
    data.to_inner_iid(yad_id):yad_id for yad_id in df["item"].tolist()
}
yad_id_to_item_id = {
    yad_id:data.to_inner_iid(yad_id) for yad_id in df["item"].tolist()

}

model = surprise.NMF(random_state=Config.seed, n_factors=240)
model.fit(data)

# %%
import torch

sample_n = train_with_label["session_id"].nunique()
ratings = torch.mm(
    torch.tensor(model.pu[:sample_n, :], dtype=torch.float16).cuda(),
    torch.tensor(model.qi.transpose(1, 0), dtype=torch.float16).cuda(),
).cpu().numpy()
ratings += model.bu.reshape(-1, 1).astype(np.float16)[:sample_n, :]
ratings += model.bi.reshape(1, -1).astype(np.float16)

# %%
# np.save(f"../features/NMF_nfactors240_float16_{Config.experiment_name}.npy", ratings)
pickle.dump(
    item_id_to_yad_id, open(f"../features/NMF_nfactors240_float16_{Config.experiment_name}.pkl", "wb")
)
pickle.dump(
    yad_id_to_item_id, open(f"../features/NMF_nfactors240_float16_{Config.experiment_name}.pkl", "wb")
)

# %%
get_top_K(ratings[0], 11)

# %%
oof_preds = []
for i, last_idx in zip(range(sample_n), train.groupby("session_id").tail(1)["yad_no"][:sample_n]):
    oof_preds.append(
        [item_id_to_yad_id[item_id] for item_id in get_top_K(ratings[i, :], 11) if item_id_to_yad_id[item_id] != last_idx]
    )


# %%
oof_preds[0]

# %%
train_with_label.loc[train_with_label["session_id"] == train_with_label["session_id"].iloc[0]]

# %%
mapk(
    [[idx] for idx in label["yad_no"].tolist()][:sample_n],
    oof_preds[:sample_n],
)

# %%
_oof_preds = []
for i in range(sample_n):
    _oof_preds.append(
        [item_id_to_yad_id[item_id] for item_id in get_top_K(ratings[i, :], 11)]
    )

mapk(
    [[idx] for idx in train.groupby("session_id").sample(1)["yad_no"].tolist()],
    _oof_preds[:sample_n],
)

# %%
_oof_preds = []
for i in range(sample_n):
    _oof_preds.append(
        [item_id_to_yad_id[item_id] for item_id in get_top_K(ratings[i, :], 11)]
    )

mapk(
    [[idx] for idx in train.groupby("session_id").head(1)["yad_no"].tolist()],
    _oof_preds[:sample_n],
)

# %% [markdown]
# - NMF(random_state=SEED, n_factors=240): 
# 

# %%
sample_n = test["session_id"].nunique()
ratings = torch.mm(
    torch.tensor(model.pu[-sample_n:, :], dtype=torch.float16).cuda(),
    torch.tensor(model.qi.transpose(1, 0), dtype=torch.float16).cuda(),
).cpu().numpy()
ratings += model.bu.reshape(-1, 1).astype(np.float16)[-sample_n:, :]
ratings += model.bi.reshape(1, -1).astype(np.float16)

# %%
sub = pd.read_csv(CSVPath.submission)

preds = []
for i, yad_id in zip(range(len(ratings)), test.groupby("session_id").tail(1)["yad_no"].tolist()):
    preds.append(
        [item_id_to_yad_id[item_id]
        for item_id in get_top_K(ratings[i, :], 11)
        if item_id_to_yad_id[item_id] != yad_id][:10]
    )


for i in range(10):
    sub[f"predict_{i}"] = np.array(preds)[:, i]

# %%
sub.to_csv(f"../sumission_{Config.experiment_name}.csv", index=False)

# %%



