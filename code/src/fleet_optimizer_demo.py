import pandas as pd
import logging
from compendium import Compendium
from fleet_optimizer import FleetOptimizer
from time import perf_counter
import pickle

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

n_pop = 25
n_gen = 25
optim = FleetOptimizer(V=V[['vehicle', 'procedure', 'latitude', 'longitude']],
                       n_pop=n_pop,
                       n_gen=n_gen,
                       c=c)

start = perf_counter()
res = optim.evaluate()
end = perf_counter()

duration = int((end - start)*1000)

with open(f"../data/pickles/fleet_optim_pop_{n_pop}_gen_{n_gen}_dur_{duration}ms.pkl", 'wb') as f:
    pickle.dump(res, f)

logger.info("Done")

# val = res.algorithm.callback.data["F_best"]
# plt.plot(np.arange(len(val)), val)
# plt.show()

# PlotVehicleLocations(V,
#                      c.facs[['name', 'latitude', 'longitude']],
#                      title="Un-optimized fleet")

# if res.X.ndim > 1:
#     res_x = res.X[0]
#     print(f"{len(res.X)} optimal solutions, showing the zeroth solution.")

# V_best = pd.DataFrame(columns=['loc'],
#                       data=res_x)

# V_best['vehicle'] = V['vehicle']
# V_best['procedure'] = V['procedure']

# i_F = np.arange(len(c.facs))
# lat_dict = (dict(zip(i_F, c.facs['latitude'])))
# lon_dict = (dict(zip(i_F, c.facs['longitude'])))

# V_best["latitude"] = V_best["loc"].map(lat_dict)
# V_best["longitude"] = V_best["loc"].map(lon_dict)

# PlotVehicleLocations(V_best,
#                      c.facs[['name', 'latitude', 'longitude']],
#                      title="Optimized fleet")
