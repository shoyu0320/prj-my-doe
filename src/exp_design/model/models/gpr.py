import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    Matern,
    WhiteKernel,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

from exp_design.model.models.base import Model


class GPRModel(Model):
    kernels = [
        DotProduct() + WhiteKernel(),
        ConstantKernel() * RBF() + WhiteKernel(),
        ConstantKernel() * RBF() + WhiteKernel() + DotProduct(),
        # ConstantKernel() * RBF(np.ones(8)) + WhiteKernel(),
        # ConstantKernel() * RBF(np.ones(8)) + WhiteKernel() + DotProduct(),
        ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + DotProduct(),
        ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + DotProduct(),
        ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + DotProduct(),
    ]

    def __init__(self, n_split: int = 10):
        self.cv = KFold(n_splits=n_split, random_state=2023, shuffle=True)
        self.best_kernel = self.kernels[0]
        self.model = None
        self.current_obj = None
        self.current_std = None
        self.obj_dims = None

    def modeling(self):
        self.model = GaussianProcessRegressor(alpha=0, kernel=self.best_kernel)

    def calc_kernel_score(self, kernel, desc, obj) -> float:
        model = GaussianProcessRegressor(alpha=0, kernel=kernel)
        score = 0
        estimated_obj = cross_val_predict(model, desc, obj, cv=self.cv)

        for obj_idx in range(obj.shape[1]):
            cur_est = estimated_obj[:, obj_idx]
            cur_obj = obj.values

            flattened_obj = np.ndarray.flatten(cur_obj)
            flattened_est = np.ndarray.flatten(cur_est)
            score += np.log(r2_score(flattened_obj, flattened_est) + 1)
        return score

    def optimize(self, descriptors: pd.DataFrame, objectives: pd.DataFrame):
        best_score = -np.inf
        best_kernel = None
        self.obj_dims = objectives.shape[1]
        size = len(self.kernels)

        for index, kernel in enumerate(self.kernels):
            print(f"\r{index + 1}/{size}", end="")
            score = self.calc_kernel_score(kernel, descriptors, objectives)
            if best_score < score:
                best_score = score
                best_kernel = kernel
        self.best_kernel = best_kernel
        self.fit(descriptors, objectives)

    def fit(self, descriptors: pd.DataFrame, objectives: pd.DataFrame):
        if self.model is None:
            self.modeling()
        self.obj_dims = objectives.shape[1]
        self.model.fit(descriptors, objectives)

    def predict(self, descriptors: pd.DataFrame):
        estimated_obj, obj_std = self.model.predict(descriptors, return_std=True)
        self.current_obj = estimated_obj
        self.current_std = obj_std
        columns = [f"estimated_{i}" for i in range(self.obj_dims)]
        return pd.DataFrame(estimated_obj, index=descriptors.index, columns=columns)
