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
        """Initialise an optimizer to optimize
        a list of facility assignments.

        Args:
            V: List of vehicles and procedures.
            n_pop (int): Optimiser population size.
            n_gen (int): Optimiser termination generation.
            c (Compendium): Static vehicle information.
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
        """Initialise the optimiser population from
        a procedures bay list.

        Args:
            V: List of vehicles and procedures.
            n_pop (int): Optimiser population size.
            flatten (bool, optional): Flatten each member. Defaults to True.

        Returns:
            Initial optimizer population.
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
        """Optimiser a solution

        Args:
            seed (int, optional): Seed the optimiser state. Defaults to 0.

        Returns:
            Object containing result and optimization data.
        """

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
