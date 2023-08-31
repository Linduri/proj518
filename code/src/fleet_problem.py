import numpy as np
import pandas as pd
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
            V (_type_): List of vehicles, procedures and locations.
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
        self.V

        super().__init__(elementwise,
                         **kwargs)

        self.logger.debug("Initialized fleet problem.")

    def _evaluate(self, x, out, *args, **kwargs):
        # Reshape to two columns...
        #  ________________________________
        # | vehicle | procedure | facility |
        # |=========|===========|==========|
        # |   int   |    int    |   int    |
        # |   ...   |    ...    |   ...    |

        _x = np.reshape(x, (-1, 3))
        X = pd.DataFrame(columns=['v', 'p', 'f'],
                         data=_x)

        F = X.groupby('f',
                      as_index=False,
                      group_keys=False)

        for f in F:
            print(f)

        # f_tups = self.c.facs.itertuples
        # for _, name, lat, lon, start, open, close, stop, bays in f_tups:
        #     self.F.append(FacilityOptimizer(_x,
        #                                     n_bays=bays,
        #                                     n_pop=self.n_pop,
        #                                     c=self.c))

        out['F'] = 1
        self.logger.debug(f"\n{out['F']}")
