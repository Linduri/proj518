import numpy as np
import pandas as pd
import math
from statistics import mean
from pymoo.core.problem import ElementwiseProblem
import logging
from compendium import Compendium
from facility_optimizer import FacilityOptimizer


class Fleet(ElementwiseProblem):
    def __init__(self,
                 F,
                 V,
                 n_pop,
                 n_gen,
                 c: Compendium,
                 elementwise=True,
                 **kwargs):
        """Initialise the fleet problem.

        Args:
            F: List of facilities.
            V: List of vehicles and facility assignments.
            n_pop (int): Optimiser population size.
            n_gen (int): Optimiser termination generation.
            c (Compendium): Static vehicle information.
            elementwise (bool, optional): Run the evaluator in series.
            Defaults to True.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initializing fleet problem...")

        self.c = c
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.F = F
        self.V = V
        self.n_var = len(self.F[0])
        self.n_cols = 1

        super().__init__(n_var=self.n_var,
                         n_obj=2,
                         xl=0,  # xl=[0, 0],
                         xu=100,  # xu=[sys.maxsize, n_bays],
                         n_ieq_constr=0,
                         elementwise=elementwise,
                         **kwargs)

        self.logger.debug("Initialized fleet problem.")

    def _mean_dist(self,
                   F: np.array) -> float:
        """Find average distance of each vehicle to
        assigned facility.

        Args:
            F (np.array): Facility assignments.

        Returns:
            float: Mean travel distance to get vehicle
            to assigned facility.
        """

        D = []
        for i, f in enumerate(F):
            f_lat = self.c.facs.iloc[f]['latitude']
            f_lon = self.c.facs.iloc[f]['longitude']
            v_lat = self.V.iloc[i]['latitude']
            v_lon = self.V.iloc[i]['longitude']
            D.append(math.dist([f_lat, f_lon], [v_lat, v_lon]))

        return mean(D)

    def _max_duration(self,
                      F: np.array) -> float:
        """Find the maximum end time of any operation
        across all facilities.

        Args:
            F: List of facility assignments.

        Returns:
            The maximum facility operation end time.
        """

        df = pd.DataFrame(columns=['f'],
                          data=F)

        F = []
        for f, vp in df.groupby('f', as_index=False, group_keys=False):
            self.logger.info(f"Optimizing facility {f}")
            V = self.V.iloc[list(vp.index.values)]
            optim = FacilityOptimizer(
                V[['vehicle', 'procedure']],
                n_bays=self.c.facs.iloc[f]['bays'],
                n_pop=self.n_pop,
                n_gen=self.n_gen,
                c=self.c
            )
            res = optim.evaluate()
            F.append(res.algorithm.callback.data["F_opt"])

        f_max = 0
        for f in F:
            arr = np.array(f)
            f_max = arr[:, 0].max() if arr[:, 0].max() > f_max else f_max

        # If no results log a huge value to maximise penalty.
        if f_max == 0:
            f_max = 999999999999999

        return f_max

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluates the fleet problem.

        Args:
            x: Population member to evaluate.

            out: Evaluation results.
        """

        # Reshape data to column of assigned facility ids.
        F = np.reshape(x, (-1, 1))

        out['F'] = [self._mean_dist(F),
                    self._max_duration(F)]
        self.logger.debug(f"\n{out['F']}")
