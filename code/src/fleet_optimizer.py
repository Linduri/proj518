import pandas as pd
import numpy as np
import logging
from compendium import Compendium
from graphing import PlotVehicleLocations
from optimizers import FleetOptimizer
import matplotlib.pyplot as plt

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
faults = V[['vehicle', 'procedure']].to_numpy()

optim = FleetOptimizer(V=V[['vehicle', 'procedure', 'latitude', 'longitude']],
                       n_pop=100,
                       n_gen=50,
                       c=c)

res = optim.evaluate()

val = res.algorithm.callback.data["F_best"]
plt.plot(np.arange(len(val)), val)
plt.show()

PlotVehicleLocations(V,
                     c.facs[['name', 'latitude', 'longitude']],
                     title="Un-optimized fleet")

V_best = pd.DataFrame(columns=['loc'],
                      data=res.X)

V_best['vehicle'] = V['vehicle']
V_best['procedure'] = V['procedure']

i_F = np.arange(len(c.facs))
lat_dict = (dict(zip(i_F, c.facs['latitude'])))
lon_dict = (dict(zip(i_F, c.facs['longitude'])))

V_best["latitude"] = V_best["loc"].map(lat_dict)
V_best["longitude"] = V_best["loc"].map(lon_dict)

PlotVehicleLocations(V_best,
                     c.facs[['name', 'latitude', 'longitude']],
                     title="Optimized fleet")
