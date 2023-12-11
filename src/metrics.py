import numpy as np

def apk(y_i_true, y_i_pred, k):
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

    return sum_precision / min(len(y_i_true), k)


def mapk(y_true, y_pred, k: int = 10):
    return np.mean([apk(y_i_true, y_i_pred, k) for y_i_true, y_i_pred in zip(y_true, y_pred)])
