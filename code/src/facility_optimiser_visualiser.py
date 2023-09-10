import numpy as np
import pickle
from statistics import mean
import matplotlib.pyplot as plt
import math
from facility_optimizer import FacilityOptimizer
from compendium import Compendium
from graphing import PlotBayOps
import pandas as pd

facilities_csv = "../data/facilities.csv"
procedure_names_csv = "../data/procedure_names.csv"
procedure_steps_csv = "../data/procedure_steps.csv"
operations_csv = "../data/operations.csv"

c = Compendium(facilities_csv,
               procedure_names_csv,
               procedure_steps_csv,
               operations_csv)

# file = '../data/pickles/facility_optim.pkl'
file = '../data/pickles/facility_optim_pop_100_gen_100_dur_896410ms.pkl'
with open(file, 'rb') as f:
    res = pickle.load(f)

# Plot objectives over time.
F_opt = np.array(res.algorithm.callback.data["F_opt"])
F = np.array(res.algorithm.callback.data["F"])

# # F0_mean = np.array([mean(f.T[0]) for f in F])
# # F0_std = np.array([np.std(f.T[0]) for f in F])
# # F0_lwr = F0_mean - F0_std
# # F0_upr = F0_mean + F0_std

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

y_labels = ['Maintenance Duration', 'Bay Utilisation']

fig, axs = plt.subplots(F_mean.shape[1],
                        1,
                        sharex=True,
                        figsize=(10, 10))
for c, f in enumerate(F_mean.T):
    x = np.arange(len(f))
    axs[c].plot(x,
                F_opt[:, c],
                color="red",
                label="Optimal")
    axs[c].plot(x,
                F_mean[:, c],
                # linestyle="dashed",
                color="black",
                label="Mean")
    axs[c].plot(x,
                F_lwr[:, c],
                linestyle="dashed",
                color="black",
                label="1 S.D")
    axs[c].plot(x,
                F_upr[:, c],
                linestyle="dashed",
                color="black")

    axs[c].scatter(F_gen[:, 0],
                   F_gen[:, c+1],
                   s=0.25,
                   color="gray")

    axs[c].set_ylabel(y_labels[c])

axs[0].legend()
plt.xlabel("Generation")
# x = np.arange(len(F), step=5)
# plt.xticks(range(len(x)), x)
plt.grid()
plt.suptitle("Facility Optimiser Evolution \nPopulation = 100")
plt.show()

arr = res.algorithm.callback.data["x_best"]
g_opt = []
for i in range(1, len(arr) - 1):
    if np.array_equiv(arr[i], arr[i+1]) is False:
        g_opt.append(i)

i_opt = range(len(g_opt))

plt.clf()
plt.plot(g_opt,
         i_opt,
         'o',
         linestyle="-",
         color='black')
plt.grid()
plt.ylabel("Optimum Number")
plt.xlabel("Generation")
plt.title("Optimum Number vs Generation")
plt.show()

diff = [g_opt[i + 1] - g_opt[i] for i in range(len(g_opt)-1)]
i_diff = [i for i, d in enumerate(diff) if d != 1]
g_diff = np.array(g_opt)[np.array(i_diff)+1]
print(g_diff)

n_pop = 100
n_gen = 100


# Load vehicle faults.
V = pd.read_csv("../data/vehicle_faults.csv")
V = V[['vehicle', 'procedure']].to_numpy()

optim = FacilityOptimizer(V,
                          n_bays=3,
                          n_pop=n_pop,
                          n_gen=n_gen,
                          c=c,
                          verbose=True)

cols = 2
fig, axs = plt.subplots(math.ceil(len(g_diff)/cols),
                        cols,
                        sharex=True,
                        figsize=(10, 10))
for c, g in enumerate(g_diff):
    x = res.algorithm.callback.data["x_best"][g]
    if x is not None:
        x = np.reshape(x, (-1, 4))
        D = pd.DataFrame(columns=['v', 'p', 'i', 'b'],
                         data=x)

        ops = optim.p.expand_ops(D)
        PlotBayOps(ops,
                   color_col='v')
    
    # for i, x in enumerate(res.algorithm.callback.data["x_best"]):
#     if x is not None:
#         print(f"Generation {i}")
#         x = np.reshape(x, (-1, 4))

#         D = pd.DataFrame(columns=['v', 'p', 'i', 'b'],
#                          data=x)

#         ops = optim.p.expand_ops(D)
#         PlotBayOps(ops,
#                    color_col='v')

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