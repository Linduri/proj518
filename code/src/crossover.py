import numpy as np
from pymoo.core.crossover import Crossover


class FleetCrossover(Crossover):
    """The crossover mechanism to cross two potential
    facility assignment lists.
    """

    def __init__(self, shift=False, prob=0.5, **kwargs):
        n_parents = 2
        n_offspring = 2
        super().__init__(n_parents,
                         n_offspring,
                         **kwargs)
        self.shift = shift

    def _do(self, problem, X, **kwargs):
        """Cross over two facility assignment lists.

        Args:
            problem: The pymoo problem being optimised.
            X: Collection of matings.

        Returns:
            Crossed matings.
        """
        _, n_matings, n_var = X.shape
        Y = np.full((self.n_offsprings,
                     n_matings,
                     n_var),
                    -1,
                    dtype=int)

        for i in range(n_matings):
            a, b = X[:, i, :]
            _a = a.reshape((-1, problem.n_cols))
            _b = b.reshape((-1, problem.n_cols))
            n = len(_a)

            if n > 1:
                # Select (S) and swap half of rows.
                rows = np.array(range(n))
                np.random.shuffle(rows)
                S = rows[:int(n/2)]
                tmp = _b[S].copy()
                _b[S], _a[S] = _a[S], tmp

                Y[0, i, :] = _a.reshape(a.shape)
                Y[1, i, :] = _b.reshape(b.shape)

            else:
                # Else swap genes.
                Y[0, i, :] = b
                Y[1, i, :] = a

        return Y


class BayCrossover(Crossover):
    """The crossover mechanism to cross two potential
    bay assignment lists.
    """

    def __init__(self, shift=False, prob=0.5, **kwargs):
        n_parents = 2
        n_offspring = 2
        super().__init__(n_parents,
                         n_offspring,
                         **kwargs)
        self.shift = shift

    def _do(self, problem, X, **kwargs):
        """Cross over two bay assignment lists.

        Args:
            problem: The pymoo problem being optimised.
            X: Collection of matings.

        Returns:
            Crossed matings.
        """
        _, n_matings, n_var = X.shape
        Y = np.full((self.n_offsprings,
                     n_matings,
                     n_var),
                    -1,
                    dtype=int)

        for i in range(n_matings):
            a, b = X[:, i, :]
            _a = a.reshape((-1, problem.n_cols))
            _b = b.reshape((-1, problem.n_cols))
            n = len(_a)

            if n > 1:
                # Select (S) and swap half of the bays
                # to the same priority.
                rows = np.array(range(n))
                np.random.shuffle(rows)
                S = rows[:int(n/2)]
                tmp = _b[S].copy()
                _b[S], _a[S] = _a[S], tmp

                Y[0, i, :] = _a.reshape(a.shape)
                Y[1, i, :] = _b.reshape(b.shape)

            else:
                # Else swap genes.
                Y[0, i, :] = b
                Y[1, i, :] = a

        return Y
