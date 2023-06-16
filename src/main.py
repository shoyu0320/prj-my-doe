import numpy as np
import pandas as pd

from exp_design.data_gen.domain.sample import Parameters
from exp_design.data_gen.searcher.d_optimal_criteria import DCriteriaParameterSearcher
from exp_design.model.area_scanner.knn import KNNScanner
from exp_design.model.models.gpr import GPRModel
from exp_design.model.sampler.ad_base_sampler import ExperimentSampler
from exp_design.model.sampler.bayes_sampler import (
    EIExperimentSampler,
    MIExperimentSampler,
    PIExperimentSampler,
    PTRExperimentSampler,
)


def fn(df):
    predesigned_beta = np.array([0.1, 1, 20, 22, 2, 3, 5, 0])

    df_target = (df.values @ predesigned_beta) ** (
        -np.log(np.abs(df.values + 1)) @ predesigned_beta
    ) + np.random.randn(len(df)) * 10
    return df_target


def main():
    raw_data = 100

    searcher = DCriteriaParameterSearcher(raw_data, 10000)
    selected, remaining = searcher.search(Parameters)

    predesigned_beta = np.array([0.1, 1, 20, 22, 2, 3, 5, 0])
    selected["target"] = fn(selected)

    remaining_target = fn(remaining)

    target_columns = "target"
    non_target_columns = [col for col in selected.columns if col != target_columns]

    pred_descriptors = remaining[non_target_columns]
    descriptors = selected[non_target_columns]
    objectives = selected[[target_columns]]

    model = GPRModel()
    ad_method = "ei"
    parameters = {"relaxation": 1e-5, "delta": 1e-6, "target_range": (300.0, 700.0)}

    if ad_method == "knn":
        ad_scanner = KNNScanner(5, metric="euclidean", ad_rate_in_train=0.95)
        sampler = ExperimentSampler(
            model, ad_scanner, normalizer="standard", parameters=parameters
        )
    elif ad_method == "mi":
        sampler = MIExperimentSampler(
            model, normalizer="standard", parameters=parameters
        )
    elif ad_method == "pi":
        sampler = PIExperimentSampler(
            model, normalizer="standard", parameters=parameters
        )
    elif ad_method == "ei":
        sampler = EIExperimentSampler(
            model, normalizer="standard", parameters=parameters
        )
    elif ad_method == "ptr":
        sampler = PTRExperimentSampler(
            model, normalizer="standard", parameters=parameters
        )
    sampler.fit(descriptors, objectives)
    sampler.set_sampler(remaining)

    output = []
    num_experiments = 50

    for i in range(num_experiments):
        print(f"\rNumber of Experiments: {i}", end="")
        new_desc = sampler.sample()
        new_obj = pd.DataFrame(fn(new_desc), columns=["target"])
        sampler.update(new_desc, new_obj)
        output.append(new_obj.values[0][0])

    best_value = max(output)
    print(f"Best Value is {best_value}")


if __name__ == "__main__":
    main()
