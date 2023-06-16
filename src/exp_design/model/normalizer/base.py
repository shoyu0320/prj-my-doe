import pandas as pd


class Normalizer:
    def forward(self, df: pd.DataFrame):
        raise NotImplementedError()

    def backward(self, df: pd.DataFrame):
        raise NotImplementedError()
