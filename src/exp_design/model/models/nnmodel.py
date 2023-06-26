from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from exp_design.model.models.base import Model


class NNModel(Model):
    nn_model = tf.keras.Sequential(
        [
            tf.keras.layers.BatchNormalization(name="bn"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    nn_model.compile(optimizer="adam", loss="mse", metrics=["mape"])

    def __init__(self, nn_model=None, params: dict[str, Any] = {}):
        self.params = {"epochs": 100, "batch_size": 126, "validation_split": 0.5}
        self.params.update(**params)
        self._nn_model = nn_model
        self.model = nn_model
        self.current_obj = None
        self.current_std = None
        self.obj_dims = None

    def modeling(self):
        if self._nn_model is None:
            self.model = self.nn_model
        else:
            self.model = self._nn_model

    def optimize(self, descriptors: pd.DataFrame, objectives: pd.DataFrame):
        self.fit(descriptors, objectives)

    def fit(self, descriptors: pd.DataFrame, objectives: pd.DataFrame):
        if self.model is None:
            self.modeling()
        self.obj_dims = objectives.shape[1]
        self.model.fit(descriptors, objectives, **self.params)

    def predict(self, descriptors: pd.DataFrame):
        estimated_obj = self.model.predict(descriptors)
        columns = [f"target_{i}" for i in range(self.obj_dims)]
        self.current_obj = pd.DataFrame(estimated_obj, columns=columns)
        self.current_std = pd.DataFrame(
            np.zeros_like(self.current_obj.values), columns=columns
        )
        columns = [f"estimated_{i}" for i in range(self.obj_dims)]
        return pd.DataFrame(estimated_obj, index=descriptors.index, columns=columns)
