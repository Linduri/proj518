import numpy as np
from pymoo.core.mutation import Mutation


class BayMutator(Mutation):
    """
    Mutates bay assignments.
    """

    def __init__(self, n_pop, prob=0.5):
        super().__init__()
        self.prob = prob
        self.n_pop = n_pop

    def _do(self, problem, X, **kwargs):
        """

        Reshape the population into an array of
        members.

        [ # Population
            [ # Member one
                [vehicle, procedure, priority],
                [vehicle, procedure, priority],
                [vehicle, procedure, priority]
            ],
            [ # Member Two
                [vehicle, procedure, priority],
                [vehicle, procedure, priority],
                [vehicle, procedure, priority]
            ]
        ]

        """

        _X = X.copy().reshape((self.n_pop, -1, 4))

        for idx, x in enumerate(_X):
            # Randomly choose a selection (S) of bays.
            P = np.random.rand(x.shape[0], 1)
            S = np.where(P[:, 0] > self.prob)

            # Mutate bays.
            np.put(x[:, 3],
                   S,
                   np.random.randint(1, problem.n_bays))

            # Randomly shuffle selected priorities (W).
            W = x[S][:, 2]
            np.random.shuffle(W)
            np.put(x[:, 2], S, W)

            _X[idx, :, :] = x

        return _X.reshape(X.shape)
