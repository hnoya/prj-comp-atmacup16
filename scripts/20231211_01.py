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
from src.memory import reduce_mem_usage

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
def get_embs(train: pd.DataFrame, test: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    #["wid_cd", "ken_cd", "lrg_cd", "sml_cd"]
    df = pd.concat([
        train[["session_id", col]],
        test[["session_id", col]]
    ]).reset_index(drop=True)
    df["rating"] = 1
    df = df.rename(
        columns={"session_id": "user", col: "item", "seq_no_fixed": "rating"}
    )
    reader = surprise.Reader(rating_scale=(0, 1))
    data = surprise.Dataset.load_from_df(
        df, reader
    ).build_full_trainset()
    unique_vals = sorted(list(set((train[col].tolist() + test[col].tolist()))))

    model = surprise.NMF(random_state=SEED, n_factors=int(np.sqrt(len(unique_vals)) + 1))
    model.fit(data)

    """
    ratings = torch.mm(
    torch.tensor(model.pu, dtype=torch.float16).cuda(),
    torch.tensor(model.qi.transpose(1, 0), dtype=torch.float16).cuda(),
    ).cpu().numpy()
    ratings += model.bu.reshape(-1, 1).astype(np.float16)
    ratings += model.bi.reshape(1, -1).astype(np.float16)
    return ratings[:len(train["session_id"].unique())], ratings[-len(test["session_id"].unique()):]
    """
    return model.pu[:len(train["session_id"].unique()), :], model.pu[-len(test["session_id"].unique()):]


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
train.head(3)

# %%
df_train = train.groupby("session_id")[
    ["seq_no", "yad_type", "total_room_cnt", "wireless_lan_flg", "onsen_flg", "kd_stn_5min", "kd_bch_5min", "kd_slp_5min", "kd_conv_walk_5min"]
].mean().reset_index()

df_test = test.groupby("session_id")[
    ["seq_no", "yad_type", "total_room_cnt", "wireless_lan_flg", "onsen_flg", "kd_stn_5min", "kd_bch_5min", "kd_slp_5min", "kd_conv_walk_5min"]
].mean().reset_index()

# %%
import warnings

for col in ["wid_cd", "ken_cd", "lrg_cd", "sml_cd"]:
    train_emb, test_emb = get_embs(train, test, col)
    for i in range(train_emb.shape[1]):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
            df_train[f"{col}_emb{i}"] = train_emb[:, i]
            df_test[f"{col}_emb{i}"] = test_emb[:, i]

# %%
df_train.head(3)

# %%
df_train = df_train.merge(label, on="session_id", how="left")

# %%
#df_train["fold"] = validate(
#    "StratifiedGroupKFold", df_train, 5,
#    groups=df_train["session_id"].tolist(), y=df_train["yad_no"].tolist(),
#)
#df_train[["session_id", "yad_no", "fold"]].to_csv("../data/train_stkgf_5folds_seed0.csv", index=False)

fold = pd.read_csv("../data/train_stkgf_5folds_seed0.csv")
df_train = df_train.merge(fold, on=["session_id", "yad_no"], how="left")

# %%
import os
import pickle

import pandas as pd
import numpy as np

import catboost
import lightgbm


def train_catboost(
    train_set: tuple[pd.DataFrame, pd.DataFrame],
    valid_set: tuple[pd.DataFrame, pd.DataFrame],
    categorical_features: list[str],
    fold: int,
    seed: int,
    output_path: str = "../models",
) -> None:
    X_train, y_train = train_set
    X_valid, y_valid = valid_set
    train_data = catboost.Pool(
        X_train, label=y_train, cat_features=categorical_features
    )
    eval_data = catboost.Pool(X_valid, label=y_valid, cat_features=categorical_features)
    # see: https://catboost.ai/en/docs/concepts/loss-functions-ranking#usage-information
    ctb_params = {
        "objective": "CrossEntropy", # "MultiClass",
        "loss_function": "CrossEntropy", # "CrossEntropy",
        'eval_metric' : 'AUC',
        "num_boost_round": 10_000,
        "early_stopping_rounds": 1000,
        "learning_rate": 0.1,
        "verbose": 1_000,
        "random_seed": seed,
        "task_type": "GPU",
        # "used_ram_limit": "32gb",
    }

    model = catboost.CatBoost(ctb_params)
    model.fit(train_data, eval_set=[eval_data], use_best_model=True, plot=False)
    pickle.dump(
        model, open(os.path.join(output_path, "ctb_fold{}.ctbmodel".format(fold)), "wb")
    )


def eval_catboost(
    X_valid: pd.DataFrame,
    fold: int,
    categorical_features: list[str],
    model_path: str = "../models",
) -> np.ndarray:
    model = pickle.load(
        open(os.path.join(model_path, "ctb_fold{}.ctbmodel".format(fold)), "rb")
    )
    y_pred = model.predict(catboost.Pool(X_valid, cat_features=categorical_features))
    return y_pred

def train_lightgbm(
    train_set: tuple[pd.DataFrame, pd.DataFrame],
    valid_set: tuple[pd.DataFrame, pd.DataFrame],
    categorical_features: list[str],
    fold: int,
    seed: int,
    output_path: str = "../models",
) -> None:
    X_train, y_train = train_set
    X_valid, y_valid = valid_set
    train_data = lightgbm.Dataset(
        X_train, label=y_train, categorical_feature=categorical_features
    )
    valid_data = lightgbm.Dataset(
        X_valid, label=y_valid, categorical_feature=categorical_features
    )
    lgb_params = {
        "objective": "binary",
        # "num_class": 13807, # int(max([max(y_train), max(y_valid)])),
        "learning_rate": 0.01,
        # "metric": "map",
        "seed": seed,
        "verbose": -1,
    }
    model = lightgbm.train(
        lgb_params,
        train_data,
        valid_sets=[train_data, valid_data],
        categorical_feature=categorical_features,
        num_boost_round=10000,
        callbacks=[
            lightgbm.early_stopping(stopping_rounds=1000, verbose=True),
            lightgbm.log_evaluation(200),
        ],
    )
    pickle.dump(
        model, open(os.path.join(output_path, "lgb_fold{}.lgbmodel".format(fold)), "wb")
    )


def eval_lightgbm(
    X_valid: pd.DataFrame, fold: int, model_path: str = "../models"
) -> np.ndarray:
    model = pickle.load(
        open(os.path.join(model_path, "lgb_fold{}.lgbmodel".format(fold)), "rb")
    )
    y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    return y_pred


# %%
NOT_USE_COL = ["session_id", "label", "fold"]

def train_folds(
    train: pd.DataFrame,
    n_fold: int,
    seed: int,
    model_type: str,
    output_path: str = "../models",
) -> None:
    for fold in range(n_fold):
        train_df, valid_df = (
            train.loc[train["fold"] != fold],
            train.loc[train["fold"] == fold],
        )
        valid_df = valid_df.loc[valid_df["label"].isin(list(set(train_df["label"])))]
        use_columns = [
            col for col in train_df.columns.tolist() if col not in NOT_USE_COL
        ]
        X_train = train_df[use_columns]
        y_train = train_df["label"]
        X_valid = valid_df[use_columns]
        y_valid = valid_df["label"]

        categorical_features = ["yad_no"]
        if model_type == "lgb":
            train_lightgbm(
                (X_train, y_train),
                (X_valid, y_valid),
                categorical_features,
                fold,
                seed,
                output_path,
            )
        elif model_type == "ctb":
            train_catboost(
                (X_train, y_train),
                (X_valid, y_valid),
                categorical_features,
                fold,
                seed,
                output_path,
            )
        elif model_type[:4] == "rec_":
            _model_type = model_type.split("rec_")[1]
            train_rec(train_df, _model_type, fold, seed, output_path)
        else:
            raise NotImplementedError(model_type)

# %%
def eval_folds(
    train: pd.DataFrame,
    n_fold: int,
    seed: int,
    model_type: str,
    model_path: str = "../models",
) -> np.ndarray:
    train["pred"] = 0
    categorical_features = ["yad_no"]
    for fold in range(n_fold):
        _, valid_df = train.loc[train["fold"] != fold], train.loc[train["fold"] == fold]
        use_columns = [
            col for col in valid_df.columns.tolist() if col not in NOT_USE_COL
        ]
        X_valid = valid_df[use_columns]
        # y_valid = valid_df["score"]
        if model_type == "lgb":
            y_pred = eval_lightgbm(X_valid, fold, model_path)
        elif model_type == "ctb":
            y_pred = eval_catboost(X_valid, fold, categorical_features, model_path)
        elif model_type[:4] == "rec_":
            _model_type = model_type.split("rec_")[1]
            y_pred = eval_rec(valid_df, fold, _model_type, model_path)
        else:
            raise NotImplementedError()
        train.loc[train["fold"] == fold, "pred"] = y_pred
    return train["pred"].values

# %%
df_train.head(3)

# %% [markdown]
# - classに閲覧したcategoryを入れ、ラベルなら1、そうでないなら0
#    - ただしそのままだと0の場合の情報が左側に入ってしまっている
#    - seq_noごとに逐次作成する必要がありそう？
#       - testも同じ条件なので気にせず良い？

# %%
train = pd.read_csv(CSVPath.log_train)
test = pd.read_csv(CSVPath.log_test)
label = pd.read_csv(CSVPath.label_train)

# %%
neg_sample = train[["session_id", "yad_no"]]
pos_sample = label[["session_id", "yad_no"]]
neg_sample["label"] = 0
pos_sample["label"] = 1
label_sample = pd.concat([neg_sample, pos_sample], axis=0).reset_index(drop=True)
df_train = df_train.drop(["yad_no"], axis=1)
df_train = df_train.merge(label_sample, on=["session_id"], how="left")
df_train.head(10)

# %%
df_train["label"].mean()

# %%
# got oom
train_folds(df_train, 5, SEED, "ctb", "../models/20231211_01_model")

# train_folds(df_train, 5, SEED, "lgb", "../models/20231211_01_model")

# %%
x_pred_ctb = eval_folds(df_train, 5, SEED, "ctb", "../models/20231211_01_model")


# %%
x_pred_ctb.shape, df_train.shape

# %% [markdown]
# - 各session_idごとにTopN（仮に10）をNMFで算出する
# - dataframeに加工し予測する
# - 各session_idごとにスコアの高い順に並べる

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

model = surprise.NMF(random_state=SEED, n_factors=480)
model.fit(data)

# %%
import torch

# %%
"""
ratings = np.dot(
    model.pu.astype(np.float16)[:sample_n, :],
    model.qi.transpose(1, 0).astype(np.float16)
)
ratings += model.bu.reshape(-1, 1).astype(np.float16)[:sample_n, :]
ratings += model.bi.reshape(1, -1).astype(np.float16)
"""
ratings = torch.mm(
    torch.tensor(model.pu, dtype=torch.float16).cuda(),
    torch.tensor(model.qi.transpose(1, 0), dtype=torch.float16).cuda(),
).cpu().numpy()
ratings += model.bu.reshape(-1, 1).astype(np.float16)
ratings += model.bi.reshape(1, -1).astype(np.float16)

# %%
# np.save("../features/NMF_nfactors480_float16_v20231211_01.npy", ratings)
# np.save("../features/NMF_nfactors480_float32_v20231211_01.npy", ratings)

# %%
oof_preds = []
for i, last_idx in zip(range(sample_n), train.groupby("session_id").tail(1)["yad_no"][:sample_n]):
    oof_preds.append(
        [item_id_to_yad_id[item_id] for item_id in get_top_K(ratings[i, :], 11) if item_id_to_yad_id[item_id] != last_idx]
    )

mapk(
    [[idx] for idx in label["yad_no"].tolist()][:sample_n],
    oof_preds[:sample_n],
)


# %% [markdown]
# - NMF(random_state=SEED, n_factors=240): 0.33048575396825397
# - NMF(random_state=SEED, n_factors=480): 0.3378028968253969
# 

# %%
ratings = np.load("../features/NMF_nfactors480_float16_v20231211_01.npy")
ratings

# %%
top_idxs = []
for rating in ratings:
    top_idx = get_top_K(rating, 10)
    top_idxs.append(top_idx.tolist())

# %%
top_idxs[:5]

# %%
import itertools

def get_flatten(l: list):
    return list(itertools.chain.from_iterable(l))

# %%
nmf_select = pd.DataFrame({
    "session_id": get_flatten(
        [[idx] * 10 for idx in train["session_id"].unique().tolist()] + [[idx] * 10 for idx in test["session_id"].unique().tolist()]
    ),
    "yad_no": get_flatten(top_idxs)
})
nmf_select["label"] = -1


# %%
train = train.merge(yado, on="yad_no", how="left")
test = test.merge(yado, on="yad_no", how="left")
df_train = train.groupby("session_id")[
    ["seq_no", "yad_type", "total_room_cnt", "wireless_lan_flg", "onsen_flg", "kd_stn_5min", "kd_bch_5min", "kd_slp_5min", "kd_conv_walk_5min"]
].mean().reset_index()

df_test = test.groupby("session_id")[
    ["seq_no", "yad_type", "total_room_cnt", "wireless_lan_flg", "onsen_flg", "kd_stn_5min", "kd_bch_5min", "kd_slp_5min", "kd_conv_walk_5min"]
].mean().reset_index()

import warnings

for col in ["wid_cd", "ken_cd", "lrg_cd", "sml_cd"]:
    train_emb, test_emb = get_embs(train, test, col)
    for i in range(train_emb.shape[1]):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
            df_train[f"{col}_emb{i}"] = train_emb[:, i]
            df_test[f"{col}_emb{i}"] = test_emb[:, i]

df_train = df_train.merge(label, on="session_id", how="left")
fold = pd.read_csv("../data/train_stkgf_5folds_seed0.csv")
df_train = df_train.merge(fold, on=["session_id", "yad_no"], how="left")

# %%
df_train = df_train.drop(["yad_no"], axis=1)
df_train = df_train.merge(nmf_select, on=["session_id"], how="left")

# %%
df_train.head(5)

# %%
x_pred_ctb = eval_folds(df_train, 5, SEED, "ctb", "../models/20231211_01_model")

# %%
oof_preds = []
for yad_idxs, scores in zip(
    df_train["yad_no"].to_numpy().reshape(-1, 10), x_pred_ctb.reshape(-1, 10)
):
    yad_rankings = get_top_K(scores, 10)
    oof_preds.append(yad_idxs[yad_rankings].tolist())

# %%
sample_n = 10_000
mapk(
    [[idx] for idx in label["yad_no"].tolist()][:sample_n],
    oof_preds[:sample_n],
)

# %%
oof_preds[:10]

# %%
[[idx] for idx in label["yad_no"].tolist()][:10]

# %%



