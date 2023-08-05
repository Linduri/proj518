from problems import Facility
import numpy as np
import pandas as pd
import logging
from compendium import Compendium
# from multiprocessing.pool import ThreadPool
# from pymoo.core.problem import StarmapParallelization
# from pymoo.algorithms.soo.nonconvex.ga import GA
# from pymoo.optimize import minimize

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

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

problem = Facility(n_bays=n_bays,
                   ops=ops)

res = dict()
print("Vehicle 0")
problem._evaluate(D.to_numpy(), res)

# print("Vehicle 1")
# problem._evaluate(V[1], res)

# print("Vehicle 2")
# problem._evaluate(V[2], res)

# # initialize the thread pool and create the runner
# n_threads = 4
# pool = ThreadPool(n_threads)
# runner = StarmapParallelization(pool.starmap)

# # define the problem by passing the starmap interface of the thread pool
# problem = Facility(elementwise_runner=runner,
#                    n_bays=V.p.max())

# res = minimize(problem, GA(), termination=("n_gen", 200), seed=1)
# print('Threads:', res.exec_time)

# pool.close()
