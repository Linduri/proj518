import numpy as np
import math
from statistics import mean
from pymoo.core.problem import ElementwiseProblem
import logging
from compendium import Compendium


class Fleet(ElementwiseProblem):
    def __init__(self,
                 V,
                 n_pop,
                 c: Compendium,
                 elementwise=True,
                 **kwargs):
        """_summary_

        Args:
            V (_type_): Vehicle information.
             ____________________________________________
            | vehicle | procedure | latitude | longitude |
            |=========|===========|==========|===========|
            |   int   |    int    |  float   |   float   |
            |   ...   |    ...    |   ...    |    ...    |

            n_pop (_type_): Size of population to evolve.
            c (Compendium): _description_
            elementwise (bool, optional): _description_. Defaults to True.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initializing fleet problem...")

        self.c = c
        self.n_pop = n_pop
        self.V = V
        self.n_var = 1

        super().__init__(n_var=self.n_var,
                         n_obj=1,
                         xl=0,  # xl=[0, 0],
                         xu=100,  # xu=[sys.maxsize, n_bays],
                         n_ieq_constr=0,
                         elementwise=elementwise,
                         **kwargs)

        self.logger.debug("Initialized fleet problem.")

    def _evaluate(self, x, out, *args, **kwargs):
        # Reshape data to column of assigned facility ids.
        F = np.reshape(x, (-1, 1))

        print(F)

        # Find average distance of each vehicle to
        # assigned facility.
        d = []
        for i, f in enumerate(F):
            f_lat = self.c.facs[f]['latitude']
            f_lon = self.c.facs[f]['longitude']
            v_lat = self.V.iloc[i]['latitude']
            v_lon = self.V.iloc[i]['longitude']
            d.append(math.dist([f_lat, f_lon], [v_lat, v_lon]))

        out['F'] = mean(d)
        self.logger.debug(f"\n{out['F']}")
