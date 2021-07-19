import pandas as pd
import numpy as np


def _split_sequences(x_seq:pd.Series, y_seq:pd.Series, window_size, pred_days):
    assert isinstance(x_seq, pd.Series) is True and isinstance(y_seq, pd.Series) is True

    x_seq = x_seq.values
    y_seq = y_seq.values
    X, y = [], []

    for i in range(len(x_seq)):
        end_index = i + window_size
        out_end_index = end_index + pred_days

        if out_end_index > len(x_seq):
            break

        seq_x, seq_y = x_seq[i:end_index], y_seq[end_index:out_end_index]

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


def _split_arrays(x_seq:pd.Series, y_seq:pd.Series, window_size, pred_days):
    assert isinstance(x_seq, pd.Series) is True and isinstance(y_seq, pd.Series) is True

    x_seq = x_seq.values
    y_seq = y_seq.values
    X, y = [], []

    for i in range(len(x_seq)):
        end_index = i + window_size
        out_end_index = end_index + pred_days

        if out_end_index > len(x_seq):
            break

        seq_x, seq_y = list(x_seq[i:end_index]), list(y_seq[end_index:out_end_index])

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.squeeze(np.array(y))
