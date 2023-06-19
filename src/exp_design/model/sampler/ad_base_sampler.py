from typing import Any

import numpy as np
import pandas as pd

from exp_design.model.area_scanner.base import ADScanner
from exp_design.model.models.base import Model
from exp_design.model.sampler.base import ExperimentSamplerBase


class ExperimentSampler(ExperimentSamplerBase):
    def __init__(
        self,
        model: Model,
        ad_scanner: ADScanner,
        normalizer: str = "standard",
        parameters: dict[str, Any] = {},
    ):
        super().__init__(model, ad_scanner, normalizer, parameters)

    def calc_sample_fn(self):
        ad_flags = self.ad_scanner(self.sampler)
        # 適用範囲外のデータ点を大きく間違った値にする
        self.sampler_est[np.logical_not(ad_flags)] = -1e16
        # Target がより大きい条件を探したい場合、次の候補としてより大きな
        sample_idx = self.sampler_est.idxmax()
        return sample_idx
