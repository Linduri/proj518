import numpy as np
from pymoo.core.mutation import Mutation


class FleetMutator(Mutation):
    """The mutation mechanism to mutate a
    facility assignment lists.
    """

    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        """Mutate a facility assignment list.

        Args:
            problem: The pymoo problem being optimised.
            X: Population.

        Returns:
            Mutated population.
        """

        _X = X.copy().reshape((-1, len(problem.V), 1))

        for idx, x in enumerate(_X):
            # Generate probability for each assignment
            # to change.
            P = np.random.rand(x.shape[0], 1)

            # Select rows past probability.
            S = np.where(P[:, 0] < self.prob)

            # Mutate bays.
            rng = np.random.default_rng()

            np.put(
                x[:, 0],
                S,
                rng.integers(0,
                             len(problem.c.facs),
                             size=len(S[0]))
            )

        return _X.reshape(X.shape)


class BayMutator(Mutation):
    """The mutation mechanism to mutate a
    bay assignment lists.
    """

    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        """Mutate a bay assignment list.

        Args:
            problem: The pymoo problem being optimised.
            X: Population.

        Returns:
            Mutated population.
        """

        _X = X.copy().reshape((-1, problem.n_rows, problem.n_cols))

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
