import pandas as pd
import surprise
import numpy as np
import warnings



def get_base_ranking_df(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train = train.groupby("session_id")[
        ["seq_no", "yad_type", "total_room_cnt", "wireless_lan_flg", "onsen_flg", "kd_stn_5min", "kd_bch_5min", "kd_slp_5min", "kd_conv_walk_5min"]
    ].mean().reset_index()

    df_test = test.groupby("session_id")[
        ["seq_no", "yad_type", "total_room_cnt", "wireless_lan_flg", "onsen_flg", "kd_stn_5min", "kd_bch_5min", "kd_slp_5min", "kd_conv_walk_5min"]
    ].mean().reset_index()
    for col in ["wid_cd", "ken_cd", "lrg_cd", "sml_cd"]:
        train_emb, test_emb = get_embs(train, test, col)
        for i in range(train_emb.shape[1]):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
                df_train[f"{col}_emb{i}"] = train_emb[:, i]
                df_test[f"{col}_emb{i}"] = test_emb[:, i]
    return df_train, df_test


def get_embs(train: pd.DataFrame, test: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
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

    model = surprise.NMF(random_state=0, n_factors=int(np.sqrt(len(unique_vals)) + 1))
    model.fit(data)
    return model.pu[:len(train["session_id"].unique()), :], model.pu[-len(test["session_id"].unique()):]
