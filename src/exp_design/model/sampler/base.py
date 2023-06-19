from typing import Any

import numpy as np
import pandas as pd

from exp_design.model.area_scanner.base import ADScanner
from exp_design.model.models.base import Model
from exp_design.model.normalizer.simple_normalizer import MinMaxNormalizer, Standardizer


class ExperimentSamplerBase:
    def __init__(
        self,
        model: Model = None,
        ad_scanner: ADScanner = None,
        normalizer: str = "standard",
        parameters: dict[str, Any] = {},
    ):
        self.model = model
        self.ad_scanner = ad_scanner
        if normalizer == "standard":
            normalizer = Standardizer
        else:
            normalizer = MinMaxNormalizer
        self.normalizer = normalizer
        self.parameters = parameters
        self.dims_lim = [1, 1]

    def _check_obj_dims(self, objectives: pd.DataFrame):
        dims = objectives.shape[1]
        assert self.dims_lim[0] <= dims <= self.dims_lim[1], (
            f"The dimension of a target in {self.__class__.__name__} must "
            f"be in the range of [{self.dims_lim[0]}, {self.dims_lim[1]}]. "
            f"But, input target has the dimensions of {dims}"
        )

    def set_normalizer(
        self, descriptors: pd.DataFrame, objectives: pd.DataFrame, update: bool = False
    ):
        self._check_obj_dims(objectives)
        self.descriptors = descriptors.copy()
        self.objectives = objectives.copy()
        if not update:
            self.desc_normalizer = self.normalizer(descriptors)
            self.obj_normalizer = self.normalizer(objectives)
        self.normd_descriptors = self.desc_normalizer.forward(descriptors)
        self.normd_objectives = self.obj_normalizer.forward(objectives)

    def has_sample(self):
        return len(self.sampler) > 10

    def fit(
        self, descriptors: pd.DataFrame, objectives: pd.DataFrame, update: bool = False
    ):
        self.set_normalizer(descriptors, objectives, update)
        if update:
            self.model.fit(self.normd_descriptors, self.normd_objectives)
        else:
            self.model.optimize(self.normd_descriptors, self.normd_objectives)

        if self.ad_scanner is not None:
            self.ad_scanner.fit(self.normd_descriptors)

    def set_sampler(self, descriptors: pd.DataFrame):
        self.sampler = self.desc_normalizer.forward(descriptors.copy())
        sampler_est = self.model.predict(self.sampler)
        self.sampler_est = self.obj_normalizer.backward(sampler_est)

    def sample(self):
        sample_idx = self.calc_sample_fn()
        sample = self.sampler.iloc[sample_idx, :]
        self.sampler = self.sampler.drop(sample_idx).reset_index().drop("index", axis=1)
        self.sampler_est = (
            self.sampler_est.drop(sample_idx).reset_index().drop("index", axis=1)
        )
        original_sample = self.desc_normalizer.backward(sample)
        return original_sample

    def update(self, descriptor: pd.DataFrame, objective: pd.DataFrame):
        descriptors = (
            pd.concat([self.descriptors, descriptor])
            .reset_index()
            .drop("index", axis=1)
        )
        objectives = (
            pd.concat([self.objectives, objective]).reset_index().drop("index", axis=1)
        )
        self.fit(descriptors, objectives, update=True)
