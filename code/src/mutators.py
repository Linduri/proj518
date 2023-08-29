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
        """

        Reshape the population into an array of
        members.

        [ # Population
            [ # Member one
                [vehicle, procedure, priority, bay],
                [vehicle, procedure, priority, bay],
                [vehicle, procedure, priority, bay]
            ],
            [ # Member Two
                [vehicle, procedure, priority, bay],
                [vehicle, procedure, priority, bay],
                [vehicle, procedure, priority, bay]
            ]
        ]

        """

        _X = X.copy().reshape((problem.n_pop, -1, problem.n_cols))

        for idx, x in enumerate(_X):
            # Randomly choose a selection (S) of bays.
            P = np.random.rand(x.shape[0], 1)
            S = np.where(P[:, 0] < self.prob)

            # Mutate bays.
            rng = np.random.default_rng()

            np.put(x[:, 3],
                   S,
                   rng.integers(1,
                                problem.n_bays + 1,
                                size=len(S[0])))

            # Randomly shuffle selected priorities (W).
            W = x[S][:, 2]
            np.random.shuffle(W)
            np.put(x[:, 2], S, W)

            _X[idx, :, :] = x

        return _X.reshape(X.shape)
