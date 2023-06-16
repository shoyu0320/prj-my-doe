import numpy as np
import pandas as pd

from exp_design.data_gen.domain.base import AbstractParameterDomain


class ParameterSearcher:
    def __init__(
        self, num_samples: int = 30, num_iter: int = 1000, verbose: bool = True
    ):
        self.num_samples = num_samples
        self.num_iter = num_iter
        self.base_sample_size = num_samples * 100
        self.verbose = verbose

    def sampling(self, num_full_samples: int):
        return np.random.choice(
            range(num_full_samples), self.num_samples, replace=False
        )

    def calc_optimal_value(self, selected_parameters: np.ndarray):
        raise NotImplementedError()

    def search(self, parameter_domain: AbstractParameterDomain):
        generated_parameters = parameter_domain.generate(self.base_sample_size)
        num_samples = generated_parameters.shape[1]
        best_optimal = 0
        best_idx = None

        for it in range(self.num_iter):
            sample_idx = self.sampling(num_samples)
            selected_parameters = generated_parameters[:, sample_idx]

            optimal_value = self.calc_optimal_value(selected_parameters)
            if optimal_value > best_optimal:
                best_optimal = optimal_value.copy()
                best_idx = sample_idx.copy()
                if self.verbose:
                    print(f"Found New Optimal Parameters! Score: {best_optimal:.2e}")
        columns = parameter_domain.columns

        remaining_idx = np.setdiff1d(np.arange(num_samples), best_idx)
        best_parameters = generated_parameters[:, best_idx]
        remaining_parameters = generated_parameters[:, remaining_idx]
        return pd.DataFrame(best_parameters.T, columns=columns), pd.DataFrame(
            remaining_parameters.T, columns=columns
        )
