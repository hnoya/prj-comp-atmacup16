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
train = pd.read_csv(CSVPath.log_train)
label = pd.read_csv(CSVPath.label_train)
yado = pd.read_csv(CSVPath.yado)

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
train = pd.concat([train, label], axis=0).reset_index(drop=True)

# %%
train.head(3)

# %%
label.head(3)

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
df = pd.concat([train[["session_id", "yad_no", "seq_no_fixed"]], test[["session_id", "yad_no", "seq_no_fixed"]]]).reset_index(drop=True)
df["seq_no_fixed"] += 1
df = df.rename(
        columns={"session_id": "user", "yad_no": "item", "seq_no_fixed": "rating"}
)

# %%
reader = surprise.Reader(rating_scale=(1, 11))
data = surprise.Dataset.load_from_df(
    df, reader
).build_full_trainset()
item_id_to_yad_id = {
    data.to_inner_iid(yad_id):yad_id for yad_id in train["yad_no"].tolist()
}

model = surprise.SVD(random_state=SEED)
model.fit(data)

# %%
ratings = np.dot(
    model.pu.astype(np.float16)[-test["session_id"].nunique():, :],
    model.qi.transpose(1, 0).astype(np.float16)[:, -test["session_id"].nunique():],
)
ratings += model.bu.reshape(-1, 1).astype(np.float16)[-test["session_id"].nunique():]
ratings += model.bi.reshape(1, -1).astype(np.float16)[-test["session_id"].nunique():]

# %%
def get_top_K(arr: np.ndarray, k: int):
    ind = np.argpartition(arr, -1 * k)[-1 * k:]
    return ind[np.argsort(arr[ind])][::-1]

# %%
[item_id_to_yad_id[item_id] for item_id in get_top_K(ratings[0, :], 11)]

# %%
preds = []
for i in range(len(ratings)):
    preds.append(
        [item_id_to_yad_id[item_id] for item_id in get_top_K(ratings[i, :], 11)]
    )

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
label.head(3)

# %%
#mapk(
#    [[idx] for idx in label["yad_no"].tolist()],
#    preds,
#)

# %% [markdown]
# oof: 0.001055416417663472
# 

# %%
sub = pd.read_csv(CSVPath.submission)
sub.head(3)

# %%
len(test), len(sub), len(ratings)

# %%
for i in range(10):
    sub[f"predict_{i}"] = np.array(preds)[:, i]

# %%
sub.to_csv("../sumission_20231210_01_eda.csv", index=False)

# %%


# %%
reader = surprise.Reader(rating_scale=(0, 1))
df["rating"] = 1
data = surprise.Dataset.load_from_df(
    df, reader
).build_full_trainset()
item_id_to_yad_id = {
    data.to_inner_iid(yad_id):yad_id for yad_id in train["yad_no"].tolist()
}

model = surprise.SVD(random_state=SEED)
model.fit(data)

# %%
ratings = np.dot(model.pu.astype(np.float16), model.qi.transpose(1, 0).astype(np.float16))
ratings += model.bu.reshape(-1, 1).astype(np.float16)
ratings += model.bi.reshape(1, -1).astype(np.float16)

# %%
import seaborn as sns

sns.kdeplot(ratings[0])

# %%
get_top_K(ratings[0], 11)

# %%
ratings[0, 11475]

# %%
oof_preds = []
for i in range(len(ratings)):
    oof_preds.append(
        [item_id_to_yad_id[item_id] for item_id in get_top_K(ratings[i, :], 11)]
    )

# %%
mapk(
    [[idx] for idx in label["yad_no"].tolist()],
    oof_preds,
)

# %%



