import pandas as pd
import numpy as np
import logging
from fleet_problem import Fleet
from mutators import FleetMutator
from crossover import FleetCrossover
from callbacks import OptimizerCallback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from compendium import Compendium


class FleetOptimizer:

    def __init__(self,
                 V: pd.DataFrame,
                 n_pop: int,
                 n_gen: int,
                 c: Compendium) -> None:
        """_summary_

        Args:
            V (pd.DataFrame): List of vehicles, procedures and locations.
             ____________________________________________
            | vehicle | procedure | latitude | longitude |
            |=========|===========|==========|===========|
            |   int   |    int    |  float   |   float   |
            |   ...   |    ...    |   ...    |    ...    |

            n_pop (_type_): Size of population to evolve.
            c (Compendium): Collection of static
            vehicle and facility data.
        """
        self.logger = logging.getLogger(__name__)

        self.V = V
        self.c = c
        self.n_pop = n_pop
        self.n_gen = n_gen

        self.logger.debug("Seeding population...")
        self.pop = self.seed_pop(V,
                                 self.n_pop,
                                 flatten=True)
        self.logger.debug("Seeded population.")

        self.logger.debug("Initializing problem...")
        self.p = Fleet(
            F=self.pop,
            V=V,
            n_pop=len(self.pop),
            n_gen=self.n_gen,
            c=self.c
        )
        self.logger.debug("Initialized problem.")

        self.logger.debug("Initializing mutator...")
        m = FleetMutator()
        self.logger.debug("Initialized mutator.")

        self.logger.debug("Initializing crossover...")
        x = FleetCrossover()
        self.logger.debug("Initialized crossover.")

        self.logger.debug("Initializing callback...")
        self.c = OptimizerCallback()
        self.logger.debug("Initialized callback.")

        self.logger.debug("Initializing algorithm...")
        self.a = NSGA2(
            pop_size=len(self.pop),
            sampling=self.pop,
            mutation=m,
            crossover=x
        )
        self.logger.debug("Initialized algorithm.")

        self.logger.debug("Initializing termination...")
        self.t = get_termination("n_gen", n_gen)
        self.logger.debug("Initialized termination.")

    def seed_pop(self,
                 V,
                 n_pop,
                 flatten=True):
        """_summary_

        Args:
            V (_type_): _description_
             ____________________________________________
            | vehicle | procedure | latitude | longitude |
            |=========|===========|==========|===========|
            |   int   |    int    |  float   |   float   |
            |   ...   |    ...    |   ...    |    ...    |

            n_bays (_type_): _description_
            n_pop (_type_): _description_
            flatten (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        pop = []
        rng = np.random.default_rng()

        for _ in range(n_pop):
            # Assign random facilities.
            F_rnd = rng.integers(0,
                                 len(self.c.facs),
                                 len(V))

            # Convert rows to columns.
            F_rnd = F_rnd.reshape((-1, 1))

            # Concatenate the facility assignments
            # to create a new population member (m)
            m = F_rnd

            if flatten is True:
                m = m.flatten()

            pop.append(m)

        return np.array(pop)

    def evaluate(self, seed=0):

        self.logger.debug("Minimizing problem...")
        res = minimize(
            problem=self.p,
            algorithm=self.a,
            termination=self.t,
            seed=seed,
            save_history=True,
            verbose=True,
            callback=self.c
         )
        self.logger.debug("Minimized problem.")

        return res
