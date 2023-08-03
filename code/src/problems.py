import sys
import numpy as np
from pymoo.core.problem import ElementwiseProblem
import logging


class Facility(ElementwiseProblem):

    def __init__(self, n_bays, ops, **kwargs):
        """Initializes the facility problem.

        Args:
            n_var (_type_): Number of decision variables.
            n_bays (_type_): Maximum number of bays in a facility,
            ops (_type_): List of operations and step number for each
                procedure.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initializing facility problem...")
        self.ops = ops
        self.n_var = 2

        super().__init__(n_var=self.n_var,
                         n_obj=1,
                         xl=[0, 0],
                         xu=[sys.maxsize, n_bays],
                         **kwargs)

        self.logger.debug("Initialized facility problem.")

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluates the facility problem.

        Args:
            x (_type_): Population member to evaluate.

            out (_type_): Evaluation results.
        """

        # Reshape to four columns...
        #  ______________________________________
        # | vehicle | procedure | priority | bay |
        # |=========|===========|==========|=====|
        # |   int   |    int    |    int   | int |
        # |   ...   |    ...    |    ...   | ... |

        x = np.reshape(x, (-1, 4))

        # Group by bay.
        B = np.unique(x[:, 3])

        for b in B:
            Z = x[x[:, 3] == b, :]

            # Cluster vehicles where vehicle id changes.
            #  ________________________________________________________
            # | vehicle | procedure | priority | bay | vehicle cluster |
            # |=========|===========|==========|=====|=================|
            # |   int   |    int    |    int   | int |       int       |
            # |   ...   |    ...    |    ...   | ... |       ...       |
            c = Z[:-1, 0] != Z[1:, 0]
            c = np.insert(c, 0, False)
            C = np.cumsum(c)
            Z = np.concatenate((Z, C.reshape(-1, 1)), axis=1)

            self.logger.info(Z)

        out['F'] = np.array([1])
