import numpy as np

from exp_design.data_gen.parameter.base import ParameterField


class IntParameterField(ParameterField):
    def __init__(
        self, pmin: int, pmax: int, seed: int = 2023, name: str = "IntParameter"
    ):
        self.pmin = pmin
        self.pmax = pmax + 1
        self.seed = seed
        self.name = name
        self.rounding = None

    def generate(self, size: int = 100):
        np.random.seed(self.seed)
        generated_parameters = np.random.randint(self.pmin, self.pmax, size)
        return generated_parameters


class FloatParameterField(ParameterField):
    def __init__(
        self,
        pmin: float,
        pmax: float,
        rounding: int | None = None,
        seed: int = 2023,
        name: str = "FloatParameter",
    ):
        self.pmin = pmin
        self.pmax = pmax
        self.seed = seed
        self.name = name
        self.rounding = rounding

    def generate(self, size: int = 100):
        np.random.seed(self.seed)
        generated_parameters = np.random.rand(size)
        generated_parameters = (
            generated_parameters * (self.pmax - self.pmin) + self.pmin
        )
        return generated_parameters


class CategoricalParameterField(IntParameterField):
    def __init__(
        self,
        categories: list[str],
        seed: int = 2023,
        name: str = "CategoricalParameter",
    ):
        self.categories = categories
        self.pmin = 0
        self.pmax = len(categories)
        self.seed = seed
        self.name = name
        self.rounding = None
