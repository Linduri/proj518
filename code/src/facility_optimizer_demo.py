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
from statistics import mean

from facility_optimizer import FacilityOptimizer

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
                          n_gen=20,
                          c=c,
                          verbose=True)
res = optim.evaluate()

# val = res.algorithm.callback.data["F_best"]

# plt.plot(np.arange(len(val)), val)
# plt.show()

# Plot objectives over time.
F_opt = np.array(res.algorithm.callback.data["F_opt"])
F = np.array(res.algorithm.callback.data["F"])

# F0_mean = np.array([mean(f.T[0]) for f in F])
# F0_std = np.array([np.std(f.T[0]) for f in F])
# F0_lwr = F0_mean - F0_std
# F0_upr = F0_mean + F0_std

n_f = F.shape[len(F.shape) - 1]
F_gen = np.empty((0, n_f+1))
for i, f in enumerate(F):
    b = np.empty((len(f), n_f+1))
    b[:, 0] = [i for _ in range(len(f.T[0]))]
    for c, _f in enumerate(f.T):
        b[:, c+1] = np.array(_f)

    F_gen = np.concatenate([F_gen, b])

F_mean = np.array([[mean(_f) for _f in f.T] for f in F])
F_std = np.array([[np.std(_f) for _f in f.T] for f in F])
F_lwr = F_mean - F_std
F_upr = F_mean + F_std

fig, axs = plt.subplots(F_mean.shape[1],
                        1,
                        sharex=True)
for c, f in enumerate(F_mean.T):
    x = np.arange(len(f))
    axs[c].plot(x,
                F_opt[:, c],
                color="red")
    axs[c].plot(x,
                F_mean[:, c],
                linestyle="dashed",
                color="black")
    axs[c].plot(x,
                F_lwr[:, c],
                linestyle=":",
                color="black")
    axs[c].plot(x,
                F_upr[:, c],
                linestyle=":",
                color="black")

    axs[c].scatter(F_gen[:, 0],
                   F_gen[:, c+1],
                   s=0.25,
                   color="black")

    axs[c].set_ylabel(f"F[{c}]")

plt.xlabel("Generation")
x = np.arange(len(F))
plt.xticks(range(len(x)), x)
# plt.ylabel("F")
plt.show()

# plt.clf()
# x = np.arange(len(F_opt))
# plt.plot(x, F_opt[:, 0])
# plt.plot(x, F_opt[:, 1])
# # if pf is not None:
# #     plt.plot(pf[:, 0], pf[:, 1], color="black", alpha=0.7)

# plt.show()


# def get_best_solution(X):
#     if X is None:
#         return False

#     if len(X.shape) > 1:
#         x = res.X[0] if X.shape[1] > 1 else res.X
#     else:
#         x = res.X

#         return x


# def print_opt(X):
#     x = get_best_solution(X)
#     if x is False:
#         return False

#     x = np.reshape(x, (-1, 4))

#     D = pd.DataFrame(columns=['v', 'p', 'i', 'b'],
#                      data=x)

#     ops = optim.p.expand_ops(D)
#     PlotBayOps(ops,
#                color_col='v')


# if print_opt(res.X) is False:
#     print("No constrained solutions found.")

# print("Best from each generation.")

# for i, x in enumerate(res.algorithm.callback.data["x_best"]):
#     if x is not None:
#         print(f"Generation {i}")
#         x = np.reshape(x, (-1, 4))

#         D = pd.DataFrame(columns=['v', 'p', 'i', 'b'],
#                          data=x)

#         ops = optim.p.expand_ops(D)
#         PlotBayOps(ops,
#                    color_col='v')
