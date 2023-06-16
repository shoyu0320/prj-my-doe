import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from exp_design.model.area_scanner.base import ADScanner


class KNNScanner(ADScanner):
    def __init__(
        self, n_neighbors: int = 5, metric="euclidean", ad_rate_in_train: float = 0.95
    ):
        self.n_neighbors = n_neighbors
        self.ad_rate_in_train = ad_rate_in_train
        self.ad_num_in_train = None
        self.thresh = None
        # ユークリッド距離で、対照データ点に近いk個のデータ点を調べる
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)

    def calc_mean_of_knn_dist(self, descriptors: pd.DataFrame):
        # 対照データ点にユークリッド距離で最も近いk個のデータ点を集める
        # 距離０である自分自身が入ってしまうため、n_neighbors には目的データ点数の +1 を入力する
        dist, idx = self.model.kneighbors(descriptors, n_neighbors=self.n_neighbors + 1)
        dist = pd.DataFrame(dist, index=descriptors.index)
        mean_dist = pd.DataFrame(
            dist.iloc[:, 1:].mean(axis=1), columns=["mean_of_knn_dist"]
        )
        return mean_dist

    def fit(self, descriptors: pd.DataFrame):
        self.ad_num_in_train = int(self.ad_rate_in_train * descriptors.shape[0]) - 1
        self.model.fit(descriptors)
        mean_dist = self.calc_mean_of_knn_dist(descriptors)
        sorted_dist = mean_dist.iloc[:, 0].sort_values(ascending=True)
        # 学習データのうち、95%点までを適用範囲にあると仮定して数値選択をする
        self.thresh = sorted_dist.iloc[self.ad_num_in_train]

    def judge(self, descriptors: pd.DataFrame):
        mean_dist = self.calc_mean_of_knn_dist(descriptors)
        return mean_dist <= self.thresh

    def __call__(self, descriptors: pd.DataFrame):
        return self.judge(descriptors)
