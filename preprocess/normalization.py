from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import preprocessing
import pandas as pd
import numpy as np


def normalize_l2(meta):
    columns = meta.columns
    meta = meta.replace([np.inf, -np.inf, 'np.nan', np.nan, '', ' '], np.nan)
    meta.fillna(meta.mean(), inplace=True)
    meta = pd.DataFrame(preprocessing.normalize(meta, norm='l2'), columns=columns)
    return meta


def normalize_standard(meta):
    scaler = StandardScaler()
    columns = meta.columns
    meta = pd.DataFrame(scaler.fit_transform(meta), columns=columns)
    return meta


def normalize_min_max(meta):
    scaler = MinMaxScaler()
    columns = meta.columns
    meta = pd.DataFrame(scaler.fit_transform(meta), columns=columns)
    return meta
