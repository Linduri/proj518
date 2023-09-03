import numpy as np
import pandas as pd
import piso
from pymoo.core.problem import ElementwiseProblem
import logging


class Facility(ElementwiseProblem):

    def __init__(self,
                 n_var,
                 n_bays,
                 n_pop,
                 n_rows,
                 ops,
                 elementwise=True,
                 **kwargs):
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
        self.n_cols = 4
        self.n_rows = n_rows
        self.n_bays = n_bays
        self.n_pop = n_pop

        super().__init__(n_var=n_var,
                         n_obj=1,
                         xl=0,  # xl=[0, 0],
                         xu=100,  # xu=[sys.maxsize, n_bays],
                         n_ieq_constr=1,
                         elementwise=elementwise,
                         **kwargs)

        self.logger.debug("Initialized facility problem.")

    def expand_ops(self, D):
        # Find adjacent vehicle procedures.
        D['c'] = D.groupby('b').cumcount()

        # Sort by bay and order
        D.sort_values(by=['b', 'c'], inplace=True)

        # Group adjacent vehicle bays (b).
        B = D.groupby('b',
                      as_index=False,
                      group_keys=False)

        # Group vehicle procedures into clusters (c) by vehicle.
        D['c'] = pd.concat([(b.v != b.v.shift()).cumsum() for _, b in B])

        # Unpack bay procedures.
        Ops = pd.DataFrame(columns=['v',
                                    'p',
                                    'i',
                                    'b',
                                    'c',
                                    'o',
                                    's',
                                    'd'])

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
            ops['t_e'] = OC.d.transform(max).where(
                ops.oc.transform(lambda x: (x != x.shift()))
                ).cumsum()

            # Fill any operation clusters with the same start
            # time for context retention.
            ops['t_e'] = ops.t_e.fillna(method='ffill')

            # Offset operations back by their own duration
            # to get their start time. Offset by 1 to avoid
            # overlapping intervals.
            ops['t_s'] = ops['t_e'] - ops['d'] + 1

            self.logger.debug(f"\n{ops}")

            Ops = pd.concat([Ops, ops])

        return Ops

    def constrain_simultaneity(self, ops: pd.DataFrame) -> bool:
        """Check to see if vehicles have simultaneous
        operations in different bays.

        Args:
            ops (pd.DataFrame): Expanded operations.

        Returns:
            bool: Constrained or unconstrained.
        """
        # Iterate through each vehicle (v).
        V = ops.groupby('v',
                        as_index=False,
                        group_keys=False)

        for _, v in V:
            # print(v)
            intervals = []
            OC = v.groupby('oc',
                           as_index=False,
                           group_keys=False)

            for _, oc in OC:

                B = oc.groupby('b',
                               as_index=False,
                               group_keys=False)

                for _, b in B:
                    head = b.head(1).iloc[0]
                    intervals.append((head.t_s, head.t_e))

            ii = pd.IntervalIndex.from_tuples(intervals)
            res = piso.adjacency_matrix(ii).any(axis=1).astype(int).values

            if res.any():
                # 1 is unconstrained.
                return 1

        # <0 is constrained.
        return -1

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

        _x = np.reshape(x, (-1, 4))

        D = pd.DataFrame(columns=['v', 'p', 'i', 'b'],
                         data=_x)

        ops = self.expand_ops(D)

        out['F'] = ops.t_e.max()
        self.logger.debug(f"\n{out['F']}")

        out['G'] = self.constrain_simultaneity(ops)
        self.logger.debug(f"\n{out['G']}")
