import pandas as pd


class Model:
    def fit(self, descriptors: pd.DataFrame, objectives: pd.DataFrame):
        raise NotImplementedError()

    def predict(self, descriptors: pd.DataFrame):
        raise NotImplementedError()
