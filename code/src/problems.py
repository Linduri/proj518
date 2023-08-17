import sys
import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
import logging


class Facility(ElementwiseProblem):

    def __init__(self, n_bays, n_cols, n_var, ops, **kwargs):
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
        self.n_cols = n_cols
        self.n_var = n_var
        self.n_bays = n_bays

        super().__init__(n_var=self.n_var,
                         n_obj=1,
                         xl=0,  # xl=[0, 0],
                         xu=100,  # xu=[sys.maxsize, n_bays],
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

            ops = pd.DataFrame(columns=['v',
                                        'p',
                                        'i',
                                        'b',
                                        'c',
                                        'o',
                                        's',
                                        'd'])
            for _, c in C:
                # Unpack bay procedure cluster operations (cO).
                for _, _v, _p, _i, _b, _c in c.itertuples():
                    o = self.ops[self.ops.p == _p]
                    o = o.drop(['p'], axis=1)
                    o.reset_index(inplace=True, drop=True)
                    t = np.array([[_v, _p, _i, _b, _c] for _ in range(len(o))])
                    t = pd.DataFrame(columns=['v', 'p', 'i', 'b', 'c'],
                                     data=t)

                    j = pd.concat([t, o],
                                  axis=1)

                    ops = pd.concat([ops, j])

            ops = ops.sort_values(['c', 's', 'o'])

            ops['oc'] = ops.o.transform(
                lambda x: (x != x.shift()).cumsum()
            )

            # Find end times
            OC = ops.groupby('oc')
            ops['t_e'] = OC.d.transform(max).cumsum().where(
                ops.o.transform(lambda x: (x != x.shift()))
                )

            # Fill any operation clusters with the same start
            # time for context retention.
            ops['t_e'] = ops.t_e.fillna(method='ffill')

            # Offset operations back by their own duration
            # to get their start time.
            ops['t_s'] = ops['t_e'] - ops['d']

            self.logger.debug(f"\n{ops}")

        out['F'] = ops.t_e.max()
        self.logger.debug(f"\n{out['F']}")
