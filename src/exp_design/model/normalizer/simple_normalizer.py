import numpy as np
import pandas as pd

from exp_design.model.normalizer.base import Normalizer


class Standardizer(Normalizer):
    def __init__(self, base_df: pd.DataFrame):
        self.mean = base_df.mean(axis=0).values
        self.std = base_df.std(axis=0).values
        zero_std_idx = np.nonzero(self.std == 0)[0]
        self.drop_columns = base_df.columns[zero_std_idx]

    def forward(self, _df: pd.DataFrame):
        df = _df.copy()
        df = (df - self.mean) / self.std

        # 標準偏差で割ることで標準化するので、標準偏差が０のデータは削除
        df = df.drop(self.drop_columns, axis=1)
        return df

    def backward(self, _df: pd.DataFrame):
        df = _df.copy()
        # 標準偏差で割ることで標準化するので、標準偏差が０のデータは削除
        df = df.drop(self.drop_columns, axis=1)
        df = df * self.std + self.mean
        return df


class MinMaxNormalizer(Normalizer):
    def __init__(self, base_df: pd.DataFrame):
        self.max = base_df.max(axis=0).values
        self.min = base_df.min(axis=0).values
        zero_minmax = np.nonzero(self.min == self.max)[0]
        self.drop_columns = base_df.columns[zero_minmax]

    def forward(self, _df: pd.DataFrame):
        df = _df.copy()
        df = (df - self.min) / (self.max - self.min)
        df = df.drop(self.drop_columns, axis=1)
        return df

    def backward(self, _df: pd.DataFrame):
        df = _df.copy()
        df = df * (self.max - self.min) + self.min
        return df
