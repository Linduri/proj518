import numpy as np
import pickle
from statistics import mean
import matplotlib.pyplot as plt
from facility_optimizer import FacilityOptimizer
from compendium import Compendium
from graphing import PlotBayOps
import pandas as pd

facilities_csv = "../data/facilities.csv"
procedure_names_csv = "../data/procedure_names.csv"
procedure_steps_csv = "../data/procedure_steps.csv"
operations_csv = "../data/operations.csv"

compendium = Compendium(facilities_csv,
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

# F_spread = np.array([
#     [
#         np.percentile(_f, 75) - np.percentile(_f, 25)for _f in f.T
#     ] for f in F
#     ])
# F_spread = np.array([[np.std(_f) for _f in f.T] for f in F])
# F_lwr = F_mean - F_spread/2
# F_upr = F_mean + F_spread/2

F_lwr = np.array([[np.percentile(_f, 25) for _f in f.T] for f in F])
F_upr = np.array([[np.percentile(_f, 75) for _f in f.T] for f in F])

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
                label="Upper/Lower Quartile")
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

diff = [g_opt[i + 1] - g_opt[i] for i in range(len(g_opt)-1)]
i_diff = [i for i, d in enumerate(diff) if d != 1]
g_diff = np.array(g_opt)[np.array(i_diff)+1]

i_opt = range(len(g_diff))

plt.clf()
plt.plot(i_opt,
         g_diff,
         'o',
         linestyle="-",
         color='black')
plt.grid()
plt.xlabel("Optimum Number")
plt.ylabel("Generation")
# plt.title("Optimum Number vs Generation")
plt.xticks(i_opt)
plt.show()

n_pop = 100
n_gen = 100

# Load vehicle faults.
V = pd.read_csv("../data/vehicle_faults.csv")
V = V[['vehicle', 'procedure']].to_numpy()

optim = FacilityOptimizer(V,
                          n_bays=3,
                          n_pop=n_pop,
                          n_gen=n_gen,
                          c=compendium,
                          verbose=True)

x_max = 0
t_e = []
for i in range(len(g_diff)):
    gen = g_diff[i-1]
    x = res.algorithm.callback.data["x_best"][gen]
    if x is not None:
        x = np.reshape(x, (-1, 4))
        D = pd.DataFrame(columns=['v', 'p', 'i', 'b'],
                         data=x)

        ops = optim.p.expand_ops(D)

        max = ops.t_e.max()
        t_e.append((gen, ops.t_e.max()))
        x_max = max if max > x_max else x_max

for i in range(len(g_diff)):
    gen = g_diff[i-1]
    x = res.algorithm.callback.data["x_best"][gen]
    if x is not None:
        print(f"Gen {gen}")
        x = np.reshape(x, (-1, 4))
        D = pd.DataFrame(columns=['v', 'p', 'i', 'b'],
                         data=x)

        ops = optim.p.expand_ops(D)
        fig = plt.figure(constrained_layout=True,
                         figsize=(10, 10))
        PlotBayOps(ops,
                   color_col='v',
                   labels=False,
                   x_max=x_max,
                   fig_in=fig)

t_e = np.array(t_e)
t_e = t_e[t_e[:, 0].argsort()]
savings = (1 - (t_e[-1, 1]/t_e[0, 1]))*100
print(f"The final solution is {savings}% shorter than the first solution.")
