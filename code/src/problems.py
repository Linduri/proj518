import sys
import numpy as np
import pandas as pd
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

        D = pd.DataFrame(columns=['v', 'p', 'i', 'b'],
                         data=x)

        # Find adjacent vehicle procedures.
        D['c'] = D.groupby('b').cumcount()

        # Sort by bay and order
        D.sort_values(by=['b', 'c'], inplace=True)

        # Group adjacent vehicle procedures (P).
        B = D.groupby('b',
                      as_index=False,
                      group_keys=False)

        D['c'] = B.apply(
                    lambda b: (b.v != b.v.shift()).cumsum()
                )

        # Unpack bay procedures.
        for _, b in B:
            C = b.groupby('c',
                          as_index=False,
                          group_keys=False)

            ops = np.empty((0, 7), int)
            for _, c in C:
                # Unpack bay procedure cluster operations (cO).
                for _, _v, _p, _i, _b, _c in c.itertuples():
                    o = self.ops[self.ops[:, 0] == _p]
                    t = np.array([[_v, _p, _i, _b, _c] for _ in range(len(o))])
                    o = np.concatenate([t, o[:, 1:]],
                                       axis=1)

                    ops = np.concatenate([ops, o],
                                         axis=0)

            Ops = pd.DataFrame(columns=['v', 'p', 'i', 'b', 'c', 'o', 's'],
                               data=ops)

            Ops = Ops.sort_values(['c', 's', 'o'])

            V = Ops.groupby('v')

            # for _, v in V:
            #     # Find adjacent vehicle operations.
            #     # v['oc'] = v.groupby('o').cumcount()
            #     _O = v.groupby('o',
            #                    as_index=False,
            #                    group_keys=False)

            #     v['oc'] = _O.apply(
            #         lambda x: (x.o != x.o.shift()).cumsum()
            #     )

            Ops['oc'] = V.o.transform(
                lambda x: (x != x.shift()).cumsum()
            )

            self.logger.info(f"\n{Ops}")

        out['F'] = np.array([1])
