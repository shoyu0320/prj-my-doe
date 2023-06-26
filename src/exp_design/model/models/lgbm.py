from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from exp_design.model.models.base import Model


class LightGBMModel(Model):
    def __init__(self, params: dict[str, Any]):
        self.params = params
        self.model = None
        self.current_obj = None
        self.current_std = None
        self.obj_dims = None

    def modeling(self):
        self.model = lgb.LGBMRegressor(**self.params)

    def optimize(self, descriptors: pd.DataFrame, objectives: pd.DataFrame):
        self.fit(descriptors, objectives)

    def fit(self, descriptors: pd.DataFrame, objectives: pd.DataFrame):
        if self.model is None:
            self.modeling()
        self.obj_dims = objectives.shape[1]
        self.model.fit(descriptors, objectives)

    def predict(self, descriptors: pd.DataFrame):
        estimated_obj = self.model.predict(descriptors)
        columns = [f"target_{i}" for i in range(self.obj_dims)]
        self.current_obj = pd.DataFrame(estimated_obj, columns=columns)
        self.current_std = pd.DataFrame(
            np.zeros_like(self.current_obj.values), columns=columns
        )
        columns = [f"estimated_{i}" for i in range(self.obj_dims)]
        return pd.DataFrame(estimated_obj, index=descriptors.index, columns=columns)
