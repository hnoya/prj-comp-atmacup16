# %%
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import surprise

# %%
import sys
sys.path.append("../")

from src.utils import seed_everything
from src.validate import validate

# %%
DATA_PATH: str = "../data"
SEED: int = 0

@dataclass
class CSVPath:
    submission: str = os.path.join(DATA_PATH, "sample_submission.csv")
    log_train: str = os.path.join(DATA_PATH, "train_log.csv")
    log_test: str = os.path.join(DATA_PATH, "test_log.csv")
    label_train: str = os.path.join(DATA_PATH, "train_label.csv")
    session_test: str = os.path.join(DATA_PATH, "test_session.csv")
    yado: str = os.path.join(DATA_PATH, "yado.csv")


seed_everything(SEED)

# %%
K = 10

def apk(y_i_true, y_i_pred):
    # y_predがK以下の長さで、要素がすべて異なることが必要
    # assert (len(y_i_pred) <= K)
    # assert (len(np.unique(y_i_pred)) == len(y_i_pred))

    sum_precision = 0.0
    num_hits = 0.0

    for i, p in enumerate(y_i_pred):
        if p in y_i_true: # 正解の場合のみ足す
            num_hits += 1
            precision = num_hits / (i + 1)
            sum_precision += precision

    return sum_precision / min(len(y_i_true), K)


# MAP@K を計算する関数
def mapk(y_true, y_pred):
    return np.mean([apk(y_i_true, y_i_pred) for y_i_true, y_i_pred in zip(y_true, y_pred)])

# %%
def get_top_K(arr: np.ndarray, k: int):
    ind = np.argpartition(arr, -1 * k)[-1 * k:]
    return ind[np.argsort(arr[ind])][::-1]

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
# train = pd.concat([train, label], axis=0).reset_index(drop=True)

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
        train[["session_id", "yad_no", "seq_no_fixed"]],
        test[["session_id", "yad_no", "seq_no_fixed"]]]
).reset_index(drop=True)
df["seq_no_fixed"] += 1
df = df.rename(
        columns={"session_id": "user", "yad_no": "item", "seq_no_fixed": "rating"}
)

# %%
reader = surprise.Reader(rating_scale=(1, 10))
df["rating"] = 1
data = surprise.Dataset.load_from_df(
    df, reader
).build_full_trainset()
item_id_to_yad_id = {
    data.to_inner_iid(yad_id):yad_id for yad_id in (train["yad_no"].tolist() + test["yad_no"].tolist())
}

model = surprise.NMF(random_state=SEED, n_factors=240)
model.fit(data)

# %%
sample_n = 10_000
ratings = np.dot(
    model.pu.astype(np.float16)[:sample_n, :],
    model.qi.transpose(1, 0).astype(np.float16)
)
ratings += model.bu.reshape(-1, 1).astype(np.float16)[:sample_n, :]
ratings += model.bi.reshape(1, -1).astype(np.float16)

# %%
get_top_K(ratings[0], 11)

# %%
oof_preds = []
for i, last_idx in zip(range(sample_n), train.groupby("session_id").tail(1)["yad_no"][:sample_n]):
    oof_preds.append(
        [item_id_to_yad_id[item_id] for item_id in get_top_K(ratings[i, :], 11) if item_id_to_yad_id[item_id] != last_idx]
    )


# %%
mapk(
    [[idx] for idx in label["yad_no"].tolist()][:sample_n],
    oof_preds[:sample_n],
)

# %% [markdown]
# - NMF(random_state=SEED, n_factors=30): 0.16822479437229437
# - NMF(random_state=SEED, n_factors=60): 0.29099016233766234
# - NMF(random_state=SEED, n_factors=120): 0.3197671031746031
# - NMF(random_state=SEED, n_factors=240): 0.33048575396825397
# 

# %%
raise NotImplementedError()

# %%
sample_n = test["session_id"].nunique()
targets = np.dot(
    model.pu.astype(np.float16)[-sample_n:, :],
    model.qi.transpose(1, 0).astype(np.float16)
)
targets += model.bu.reshape(-1, 1).astype(np.float16)[-sample_n:, :]
targets += model.bi.reshape(1, -1).astype(np.float16)

# %%
sub = pd.read_csv(CSVPath.submission)

preds = []
for i, yad_id in zip(range(len(targets)), test.groupby("session_id").tail(1)["yad_no"].tolist()):
    preds.append(
        [item_id_to_yad_id[item_id] for item_id in get_top_K(targets[i, :], 11) if item_id_to_yad_id[item_id] != yad_id][:10]
    )


for i in range(10):
    sub[f"predict_{i}"] = np.array(preds)[:, i]

# %%
sub.to_csv("../sumission_20231210_03.csv", index=False)

# %%



