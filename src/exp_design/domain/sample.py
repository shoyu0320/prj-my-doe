from exp_design.domain.base import AbstractParameterDomain


class Parameters(AbstractParameterDomain):
    component1 = FloatParameterField(0.5, 1.0, name="component1", seed=1001, rounding=4)
    component2 = FloatParameterField(0.0, 1.0, name="component2", seed=1002, rounding=4)
    component3 = FloatParameterField(0.0, 1.0, name="component3", seed=1003, rounding=4)
    additive1 = FloatParameterField(0, 1, name="additive1", seed=2001, rounding=4)
    additive2 = FloatParameterField(0, 1, name="additive2", seed=2002, rounding=4)
    additive3 = FloatParameterField(0, 1, name="additive3", seed=2003, rounding=4)
    temperature = IntParameterField(50, 150, name="temperature", seed=3001)
    catalyst = CategoricalParameterField(
        ["PPhos", "XPhos", "SPhos"], name="catalyst", seed=4001
    )

    groups = {
        "components": [f"component{i}" for i in range(1, 4)],
        "additive2": [f"additive{i}" for i in range(1, 4)],
    }
