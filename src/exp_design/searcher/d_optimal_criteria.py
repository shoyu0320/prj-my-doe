import numpy as np

from exp_design.searcher.base import ParameterSearcher


class DCriteriaParameterSearcher(ParameterSearcher):
    def calc_optimal_value(self, selected_parameters: np.ndarray):
        xt_x = selected_parameters @ selected_parameters.T
        det = np.linalg.det(xt_x)
        return det
