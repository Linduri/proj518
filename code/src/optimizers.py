import pandas as pd
import numpy as np
import logging
from problems import Facility
from mutators import BayMutator
from crossover import BayCrossover
from callbacks import FacilityCallback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from compendium import Compendium


class FacilityOptimizer:

    def __init__(self,
                 V: np.array,
                 n_bays: int,
                 n_pop: int,
                 c: Compendium) -> None:
        """_summary_

        Args:
            V (np.array): List of vehicle id against
            procedure id.
            c (Compendium): Collection of static
            vehicle and facility data.
        """
        self.logger = logging.getLogger(__name__)

        self.logger.debug("Seeding population...")
        self.pop = self.seed_pop(V,
                                 n_bays,
                                 n_pop,
                                 flatten=True)
        self.logger.debug("Seeded population.")

        # Each population member should have the same
        # vehicle and procedure numbers so use the first
        # member to unpack the operations to save
        # processing time on each evaluation.
        V0 = np.reshape(self.pop[0], (-1, 4))
        D = pd.DataFrame(columns=['v', 'p', 'i', 'b'],
                         data=V0)

        # Unpack procedures to get operations and steps.
        #  __________________________________________
        # | procedure | operation | step  | duration |
        # |    (p)    |    (o)    |  (s)  |    (d)   |
        # |===========|===========|=======|==========|
        # |    int    |    int    |  int  |    int   |
        # |    ...    |    ...    |  ...  |    ...   |
        self.logger.debug("Unpacking procedures...")
        P = D.p.unique()
        od_dict = c.ops.set_index('id').to_dict()['duration']
        ops = pd.DataFrame(columns=['p', 'o', 's', 'd'])
        for p in P:
            s = c.steps[c.steps.procedure == p].copy()
            s['duration'] = s.operation.replace(od_dict)
            s.rename(columns={'procedure': 'p',
                              'operation': 'o',
                              'step': 's',
                              'duration': 'd'},
                     inplace=True)

            ops = pd.concat([ops, s],
                            ignore_index=True)

        self.ops = ops
        self.logger.debug("Unpacked procedures.")

        n_bays = D.b.max()

        self.p = Facility(
            n_var=len(self.pop[0]),
            n_bays=n_bays,
            n_pop=n_pop,
            ops=ops
        )

        self.logger.debug("Initializing mutator...")
        m = BayMutator()
        self.logger.debug("Initialized mutator.")

        self.logger.debug("Initializing crossover...")
        x = BayCrossover()
        self.logger.debug("Initialized crossover.")

        self.logger.debug("Initializing callback...")
        self.c = FacilityCallback()
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
        self.t = get_termination("n_gen", 50)
        self.logger.debug("Initialized termination.")

    def seed_pop(self, V, n_bays, n_pop, flatten=True):
        pop = []

        # Generate sequential base list of priorities (P)
        P = np.arange(len(V))

        for i in range(n_pop):
            # Randomize priorities
            np.random.shuffle(P)

            # Generate randomized list of bays (B).
            rng = np.random.default_rng()
            B = rng.integers(low=1, high=n_bays+1, size=len(V))

            # Convert rows to columns.
            Pc = P.reshape((-1, 1))
            B = B.reshape((-1, 1))

            # Concatenate the new columns to the base vehicle
            # data to create a new population member (m)
            m = np.c_[V, Pc, B]

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
