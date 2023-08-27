from problems import Facility
from mutators import BayMutator
import numpy as np
import pandas as pd
import logging
from compendium import Compendium
from pymoo.algorithms.moo.nsga2 import NSGA2
from crossover import BayCrossover
from callbacks import FacilityCallback
from graphing import PlotBayOps
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import matplotlib.pyplot as plt
# from multiprocessing.pool import ThreadPool
# from pymoo.core.problem import StarmapParallelization
# from pymoo.algorithms.soo.nonconvex.ga import GA
# from pymoo.optimize import minimize

from optimizers import FacilityOptimizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

facilities_csv = "../data/facilities.csv"
procedure_names_csv = "../data/procedure_names.csv"
procedure_steps_csv = "../data/procedure_steps.csv"
operations_csv = "../data/operations.csv"

c = Compendium(facilities_csv,
               procedure_names_csv,
               procedure_steps_csv,
               operations_csv)

# Columns...
#  ______________________________________
# | vehicle | procedure | priority | bay |
# |=========|===========|==========|=====|
# |   int   |    int    |    int   | int |
# |   ...   |    ...    |    ...   | ... |
V0 = np.array([[0, 3, 0, 1],
              [0, 4, 1, 3],
              [1, 1, 2, 3],
              [1, 2, 3, 3],
              [1, 4, 4, 2],
              [2, 5, 5, 3]])

V1 = np.array([[0, 3, 5, 1],
              [0, 4, 4, 2],
              [1, 1, 3, 3],
              [1, 2, 2, 3],
              [1, 4, 1, 2],
              [2, 5, 0, 1]])

V2 = np.array([[0, 3, 3, 1],
              [0, 4, 4, 2],
              [1, 1, 1, 3],
              [1, 2, 2, 3],
              [1, 4, 5, 2],
              [2, 5, 0, 1]])

V = np.vstack((V0.flatten(),
              V1.flatten(),
              V2.flatten()))

optim = FacilityOptimizer(V, c)
res = optim.evaluate()

val = res.algorithm.callback.data["F_best"]
plt.plot(np.arange(len(val)), val)
plt.show()

print(res.X)


def print_opt(X):
    if X is None:
        return False

    x = res.X[0] if X.shape[1] > 1 else res.X

    print(x)
    x = np.reshape(x, (-1, 4))

    D = pd.DataFrame(columns=['v', 'p', 'i', 'b'],
                     data=x)

    ops = optim.p.expand_ops(D)
    PlotBayOps(ops,
               color_col='v')


if print_opt(res.X) is False:
    print("No constrained solutions found.")
