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

    def modeling(self):
        self.model = GaussianProcessRegressor(alpha=0, kernel=self.best_kernel)

    def calc_kernel_score(self, kernel, desc, obj) -> float:
        model = GaussianProcessRegressor(alpha=0, kernel=kernel)
        estimated_obj = cross_val_predict(model, desc, obj, cv=self.cv)
        flattened_obj = np.ndarray.flatten(estimated_obj)
        score = r2_score(obj, flattened_obj)
        return score

    def optimize(self, descriptors: pd.DataFrame, objectives: pd.DataFrame):
        best_score = 0
        best_kernel = None
        size = len(self.kernels)

        for index, kernel in enumerate(self.kernels):
            print(f"\r{index + 1}/{size}", end="")
            score = self.calc_kernel_score(kernel, descriptors, objectives)
            if best_score > score:
                best_score = score
                best_kernel = kernel
        self.best_kernel = best_kernel
        self.fit(descriptors, objectives)

    def fit(self, descriptors: pd.DataFrame, objectives: pd.DataFrame):
        if self.model is None:
            self.modeling()
        self.model.fit(descriptors, objectives)

    def predict(self, descriptors: pd.DataFrame):
        estimated_obj, obj_std = self.model.predict(descriptors, return_std=True)
        self.current_obj = estimated_obj
        self.current_std = obj_std
        return pd.DataFrame(
            estimated_obj, index=descriptors.index, columns=["estimated"]
        )
