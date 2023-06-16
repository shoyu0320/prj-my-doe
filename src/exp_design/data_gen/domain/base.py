import numpy as np

from exp_design.data_gen.parameter.base import ParameterField


class AbstractParameterDomain:
    groups = {}

    @classmethod
    @property
    def columns(cls):
        return list(cls.parameters.keys())

    @classmethod
    @property
    def parameters(cls):
        output = {}
        for name, field in cls.__dict__.items():
            if isinstance(field, ParameterField):
                output[name] = field
        return output

    @classmethod
    def get_validator(cls, array: np.ndarray, member_idx: list[int]):
        generators: list[ParameterField] = list(cls.parameters.values())
        pmins = np.array([generators[idx].pmin for idx in member_idx]).reshape(-1, 1)
        pmaxs = np.array([generators[idx].pmax for idx in member_idx]).reshape(-1, 1)

        min_validator = array > pmins
        max_validator = array < pmaxs
        member_validators = np.logical_and(min_validator, max_validator)
        validator = member_validators.sum(axis=0) == len(member_idx)
        return validator

    @classmethod
    def grouping(cls, generated_parameters: np.ndarray):
        names: list[str] = list(cls.parameters.keys())
        for group_name, members in cls.groups.items():
            member_idx = [names.index(m) for m in members]
            group_array = generated_parameters[member_idx, :]
            modified_group_array = group_array / group_array.sum(axis=0)

            validator = cls.get_validator(modified_group_array, member_idx)
            modified_group_array = modified_group_array[:, validator]
            generated_parameters = generated_parameters[:, validator]
            generated_parameters[member_idx, :] = modified_group_array
        return generated_parameters

    @classmethod
    def rounding(cls, generated_parameters: np.ndarray):
        for num, generator in enumerate(cls.parameters.values()):
            if generator.rounding is not None:
                generated_parameters[num] = np.round(
                    generated_parameters[num], generator.rounding
                )
        return generated_parameters

    @classmethod
    def generate(cls, size: int):
        generated_parameters = []
        for name, generator in cls.parameters.items():
            generated_parameter = generator.generate(size)
            generated_parameters.append(generated_parameter)
        generated_parameters = np.array(generated_parameters)
        generated_parameters = cls.grouping(generated_parameters)
        generated_parameters = cls.rounding(generated_parameters)
        return generated_parameters
