class EnvGenerator(object):
    def __init__(self, klass, invariants):
        self.klass = klass
        self.invariants = invariants
        self.variables = {param[0]: param[1] for param in klass.get_params() if param[0] not in invariants.keys()}

    def gen_samples(self, training=True):
        while True:
            yield self.klass(**self.invariants, **self.variables)


