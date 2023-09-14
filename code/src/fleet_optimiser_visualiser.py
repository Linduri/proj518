import numpy as np
import pickle
from graphing import PlotVehicleLocations
import matplotlib.pyplot as plt
import pandas as pd
from compendium import Compendium
from statistics import mean


facilities_csv = "../data/facilities.csv"
procedure_names_csv = "../data/procedure_names.csv"
procedure_steps_csv = "../data/procedure_steps.csv"
operations_csv = "../data/operations.csv"

compendium = Compendium(facilities_csv,
                        procedure_names_csv,
                        procedure_steps_csv,
                        operations_csv)

file = '../data/pickles/fleet_optim_pop_25_gen_25_dur_86968398ms.pkl'
with open(file, 'rb') as f:
    res = pickle.load(f)

# Plot objectives over time.
F_opt = np.array(res.algorithm.callback.data["F_opt"])
F = np.array(res.algorithm.callback.data["F"])

n_f = F.shape[len(F.shape) - 1]
F_gen = np.empty((0, n_f+1))
for i, f in enumerate(F):
    b = np.empty((len(f), n_f+1))
    b[:, 0] = [i for _ in range(len(f.T[0]))]
    for c, _f in enumerate(f.T):
        b[:, c+1] = np.array(_f)

    F_gen = np.concatenate([F_gen, b])

F_mean = np.array([[mean(_f) for _f in f.T] for f in F])


F_lwr = np.array([[np.percentile(_f, 25) for _f in f.T] for f in F])
F_upr = np.array([[np.percentile(_f, 75) for _f in f.T] for f in F])

y_labels = ['Travel Distance', 'Maintenance Duration']

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
    axs[c].grid()

    axs[c].legend()
plt.xlabel("Generation")

plt.suptitle("Population = 25")
plt.show()

V = pd.read_csv("../data/vehicle_faults.csv")
faults = V[['vehicle', 'procedure']].to_numpy()

x_lim = [50, 55]
y_lim = [-4.5, 1]
print("Un-optimised fleet")
PlotVehicleLocations(V,
                     compendium.facs[['name', 'latitude', 'longitude']],
                     # title="Un-optimized fleet",
                     x_lim=x_lim,
                     y_lim=y_lim)

if res.X.ndim > 1:
    res_x = res.X[0]
    print(f"{len(res.X)} optimal solutions, showing the zeroth solution.")

V_best = pd.DataFrame(columns=['loc'],
                      data=res_x)

V_best['vehicle'] = V['vehicle']
V_best['procedure'] = V['procedure']

i_F = np.arange(len(compendium.facs))
lat_dict = (dict(zip(i_F, compendium.facs['latitude'])))
lon_dict = (dict(zip(i_F, compendium.facs['longitude'])))

V_best["latitude"] = V_best["loc"].map(lat_dict)
V_best["longitude"] = V_best["loc"].map(lon_dict)

print("Optimised fleet")
PlotVehicleLocations(V_best,
                     compendium.facs[['name', 'latitude', 'longitude']],
                     #  title="Optimized fleet",
                     x_lim=x_lim,
                     y_lim=y_lim)
