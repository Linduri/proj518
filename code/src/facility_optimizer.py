from problems import Facility
import numpy as np
import logging
# from multiprocessing.pool import ThreadPool
# from pymoo.core.problem import StarmapParallelization
# from pymoo.algorithms.soo.nonconvex.ga import GA
# from pymoo.optimize import minimize

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

V0 = np.array([[0, 0, 1],
              [1, 1, 2],
              [2, 2, 3],
              [3, 3, 3],
              [4, 4, 2],
              [5, 5, 1]])

V1 = np.array([[0, 0, 2],
              [1, 1, 2],
              [2, 2, 2],
              [3, 3, 2],
              [4, 4, 1],
              [5, 5, 3]])

V2 = np.array([[0, 0, 2],
              [1, 1, 2],
              [2, 2, 2],
              [3, 3, 1],
              [4, 4, 1],
              [5, 5, 3]])

V = np.array([V0.flatten(),
              V1.flatten(),
              V2.flatten()])

print("Raw data")
print(V)
n_var = 2
n_bays = 3

problem = Facility(n_var=n_var,
                   n_bays=n_bays)

res = dict()
print("Vehicle 0")
problem._evaluate(V[0], res)

print("Vehicle 1")
problem._evaluate(V[1], res)

print("Vehicle 2")
problem._evaluate(V[2], res)

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
