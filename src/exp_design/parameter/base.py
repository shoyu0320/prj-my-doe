class ParameterField:
    def __init__(self):
        self.rounding = None

    def generate(self, size: int = 100):
        raise NotImplementedError()
