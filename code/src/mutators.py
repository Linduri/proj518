import numpy as np
from pymoo.core.mutation import Mutation


class BayMutator(Mutation):
    """
    Mutates bay assignments.
    """

    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):

        # Reshape to four columns...
        #  ______________________________________
        # | vehicle | procedure | priority | bay |
        # |=========|===========|==========|=====|
        # |   int   |    int    |    int   | int |
        # |   ...   |    ...    |    ...   | ... |
        _X = X.copy().reshape((-1, 4))

        # Randomly mutate a selection (S) of bays.
        P = np.random.rand(_X.shape[0], 1)
        _X[P[:, 0] > self.prob, 3] = np.random.randint(0,
                                                       problem.n_bays)

        return _X.reshape(X.shape)
