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

D = pd.DataFrame(columns=['v', 'p', 'i', 'b'],
                 data=V0)

# Unpack procedures to get operations and steps.
#  __________________________________________
# | procedure | operation | step  | duration |
# |    (p)    |    (o)    |  (s)  |    (d)   |
# |===========|===========|=======|==========|
# |    int    |    int    |  int  |    int   |
# |    ...    |    ...    |  ...  |    ...   |
logger.info("Unpacking procedures...")
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

logger.info("Unpacked procedures.")

n_var = 2
n_bays = 3

p = Facility(n_bays=n_bays,
             ops=ops,
             n_cols=4,
             n_var=len(V[0]))

# res = dict()
# print("Vehicle 0")
# p._evaluate(D.to_numpy(), res)

logger.info("Initializing mutator...")
m = BayMutator(n_pop=len(V))
logger.info("Initialized mutator.")

logger.info("Initializing crossover...")
x = BayCrossover(n_pop=len(V))
logger.info("Initialized crossover.")

logger.info("Initializing callback...")
c = FacilityCallback()
logger.info("Initialized callback.")

logger.info("Initializing algorithm...")
a = NSGA2(
    pop_size=len(V),
    sampling=V,
    mutation=m,
    crossover=x
)
logger.info("Initialized algorithm.")

logger.info("Initializing termination...")
t = get_termination("n_gen", 10)
logger.info("Initialized termination.")

logger.info("Minimizing problem...")
res = minimize(problem=p,
               algorithm=a,
               termination=t,
               # seed=_seed,
               save_history=True,
               verbose=True,
               callback=c
               )

val = res.algorithm.callback.data["F_best"]
plt.plot(np.arange(len(val)), val)
plt.show()

print(res.X)
print(res.G)

if res.X is not None:
    n_res = len(res.X)
    if n_res > 1:
        x = res.X[0]
    else:
        x = res.X

    x = np.reshape(x, (-1, 4))
    print(x)

    D = pd.DataFrame(columns=['v', 'p', 'i', 'b'],
                     data=x)

    ops = p.expand_ops(D)
    PlotBayOps(ops,
               color_col='v')

else:
    print("No constrained solutions found.")
