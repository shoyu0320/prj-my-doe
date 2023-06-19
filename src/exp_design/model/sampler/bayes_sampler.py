from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from exp_design.model.models.base import Model
from exp_design.model.sampler.ad_base_sampler import ExperimentSampler
from exp_design.model.sampler.base import ExperimentSamplerBase


class MIExperimentSampler(ExperimentSamplerBase):
    def __init__(
        self,
        model: Model,
        normalizer: str = "standard",
        parameters: dict[str, Any] = {},
    ):
        super().__init__(model, None, normalizer, parameters)
        self.delta = self.parameters.get("delta", 1e-6)
        self.alpha = np.sqrt(np.log(2 / self.delta))
        self.thresh = 0

    def calc_sample_fn(self):
        self.model.predict(self.sampler)

        std = self.model.current_std * self.obj_normalizer.std
        pred = self.obj_normalizer.backward(self.model.current_obj)

        gamma_t = np.zeros_like(pred)
        phi_t = self.alpha * (np.sqrt(std**2 + gamma_t) - np.sqrt(gamma_t))
        mi = pred + phi_t
        mi[std <= self.thresh] = 0

        mi = pd.DataFrame(mi, index=self.sampler.index)
        sample_idx = mi.idxmax()
        return sample_idx


class PIExperimentSampler(ExperimentSamplerBase):
    def __init__(
        self,
        model: Model,
        normalizer: str = "standard",
        parameters: dict[str, Any] = {},
    ):
        super().__init__(model, None, normalizer, parameters)
        self.relaxation = self.parameters.get("relaxation", 1e-2)
        self.dims_lim = [1, np.inf]

    def calc_prob_of_improve(self):
        self.model.predict(self.sampler)

        std = self.model.current_std * self.obj_normalizer.std
        pred = self.obj_normalizer.backward(self.model.current_obj)

        y_max = self.objectives.values.max(axis=0)
        delta_y = pred - y_max - self.relaxation * self.obj_normalizer.std
        prob = norm.cdf(delta_y / self.obj_normalizer.std)
        log_prob = np.log(prob + 1)
        return np.sum(log_prob, axis=1)

    def calc_sample_fn(self):
        pi = self.calc_prob_of_improve()
        pi = pd.DataFrame(pi, index=self.sampler.index)
        sample_idx = pi.idxmax()
        return sample_idx


class EIExperimentSampler(PIExperimentSampler):
    def calc_exp_prob_of_improve(self):
        self.model.predict(self.sampler)

        std = self.model.current_std * self.obj_normalizer.std
        pred = self.obj_normalizer.backward(self.model.current_obj)

        y_max = self.objectives.values.max()
        delta_y = pred - y_max - self.relaxation * self.obj_normalizer.std
        pi = norm.cdf(delta_y / self.obj_normalizer.std)
        first_item = delta_y * pi
        second_item = std * norm.pdf(delta_y)
        return first_item + second_item

    def calc_sample_fn(self):
        ei = self.calc_exp_prob_of_improve()
        ei = pd.DataFrame(ei, index=self.sampler.index)
        sample_idx = ei.idxmax()
        return sample_idx


class PTRExperimentSampler(ExperimentSampler):
    def __init__(
        self,
        model: Model,
        normalizer: str = "standard",
        parameters: dict[str, Any] = {},
    ):
        super().__init__(model, None, normalizer, parameters)
        self.target_range = self.parameters.get("target_range", [(750.0, 800.0)])
        self.dims_lim = [1, np.inf]

    def calc_prob_of_target_range(self):
        self.model.predict(self.sampler)

        std = self.model.current_std * self.obj_normalizer.std
        pred = self.obj_normalizer.backward(self.model.current_obj)

        scores = np.zeros_like(pred[:, 0])
        for obj_idx in range(self.model.obj_idx):
            target_range = self.target_range[obj_idx]
            obj_std = std[obj_idx]
            obj_pred = pred.iloc[:, obj_idx]
            y_upper = norm.cdf(target_range[1], loc=obj_pred, scale=obj_std)
            y_lower = norm.cdf(target_range[0], loc=obj_pred, scale=obj_std)
            prob = y_upper - y_lower
            log_prob = np.log(prob + 1)
            scores += log_prob
        return scores

    def calc_sample_fn(self):
        ptr = self.calc_prob_of_target_range()
        ptr = pd.DataFrame(ptr, index=self.sampler.index)
        sample_idx = ptr.idxmax()
        return sample_idx
