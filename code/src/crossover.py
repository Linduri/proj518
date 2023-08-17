import numpy as np
from pymoo.core.crossover import Crossover


class BayCrossover(Crossover):

    def __init__(self,  n_pop, shift=False, prob=0.5, **kwargs):
        self.n_pop = n_pop
        n_parents = 2
        n_offspring = 2
        super().__init__(n_parents,
                         n_offspring,
                         **kwargs)
        self.shift = shift

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

        _X = X.copy().reshape((self.n_pop, -1, problem.n_cols))

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
