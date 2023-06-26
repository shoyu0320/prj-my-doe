from typing import Any

import numpy as np
import pandas as pd

from exp_design.model.models.base import Model


class EnsembleModel(Model):
    def __init__(self, models: list[Model]):
        self.model: list[Model] = models
        self.current_obj = None
        self.current_std = None
        self.obj_dims = None

    def modeling(self):
        for model in self.model:
            model.modeling()

    def optimize(self, descriptors: pd.DataFrame, objectives: pd.DataFrame):
        for model in self.model:
            model.optimize(descriptors, objectives)

    def fit(self, descriptors: pd.DataFrame, objectives: pd.DataFrame):
        if self.model is None:
            self.modeling()
        self.obj_dims = objectives.shape[1]
        for model in self.model:
            model.fit(descriptors, objectives)

    def predict(self, descriptors: pd.DataFrame):
        estimated_obj = np.zeros_like((len(descriptors), self.obj_dims))
        for model in self.model:
            estimated_obj += model.predict(descriptors).values

        columns = [f"target_{i}" for i in range(self.obj_dims)]
        self.current_obj = pd.DataFrame(estimated_obj, columns=columns)
        self.current_std = pd.DataFrame(
            np.zeros_like(self.current_obj.values), columns=columns
        )
        columns = [f"estimated_{i}" for i in range(self.obj_dims)]
        return pd.DataFrame(estimated_obj, index=descriptors.index, columns=columns)
