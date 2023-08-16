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

        # Randomly choose a selection (S) of bays.
        P = np.random.rand(_X.shape[0], 1)
        S = np.where(P[:, 0] > self.prob)

        # Mutate bays.
        _X[S][:, 3] = np.random.randint(0,
                                        problem.n_bays)

        # Randomly shuffle selected priorities (W).
        W = _X[S][:, 2]
        np.random.shuffle(W)
        np.put(_X[:, 2], S, W)

        return _X.reshape(X.shape)
