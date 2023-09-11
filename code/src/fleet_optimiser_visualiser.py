import numpy as np
import pickle
from graphing import PlotVehicleLocations
import matplotlib.pyplot as plt
import pandas as pd
from compendium import Compendium


facilities_csv = "../data/facilities.csv"
procedure_names_csv = "../data/procedure_names.csv"
procedure_steps_csv = "../data/procedure_steps.csv"
operations_csv = "../data/operations.csv"

compendium = Compendium(facilities_csv,
                        procedure_names_csv,
                        procedure_steps_csv,
                        operations_csv)

file = '../data/pickles/fleet_optim_pop_10_gen_5_dur_590338ms.pkl'
with open(file, 'rb') as f:
    res = pickle.load(f)

val = res.algorithm.callback.data["F_opt"]
plt.plot(np.arange(len(val)), val)
plt.show()

V = pd.read_csv("../data/vehicle_faults.csv")
faults = V[['vehicle', 'procedure']].to_numpy()

PlotVehicleLocations(V,
                     compendium.facs[['name', 'latitude', 'longitude']],
                     title="Un-optimized fleet")

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

PlotVehicleLocations(V_best,
                     compendium.facs[['name', 'latitude', 'longitude']],
                     title="Optimized fleet")
