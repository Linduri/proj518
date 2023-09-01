import numpy as np
import pandas as pd
import logging
from compendium import Compendium
from graphing import PlotBayOps
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

# Load vehicle faults.
V = pd.read_csv("../data/vehicle_faults.csv")
V = V[['vehicle', 'procedure']].to_numpy()

# Columns...
#  ______________________
# | vehicle | procedure |
# |=========|===========|
# |   int   |    int    |
# |   ...   |    ...    |
# V = np.array([[0, 3],
#               [0, 4],
#               [1, 1],
#               [1, 2],
#               [1, 4],
#               [2, 5]])

optim = FacilityOptimizer(V,
                          n_bays=3,
                          n_pop=50,
                          n_gen=50,
                          c=c)
res = optim.evaluate()

val = res.algorithm.callback.data["F_best"]
plt.plot(np.arange(len(val)), val)
plt.show()


def print_opt(X):
    if X is None:
        return False

    if len(X.shape) > 1:
        x = res.X[0] if X.shape[1] > 1 else res.X
    else:
        x = res.X

    x = np.reshape(x, (-1, 4))

    D = pd.DataFrame(columns=['v', 'p', 'i', 'b'],
                     data=x)

    ops = optim.p.expand_ops(D)
    PlotBayOps(ops,
               color_col='v')


if print_opt(res.X) is False:
    print("No constrained solutions found.")
