import pandas as pd
import logging
from compendium import Compendium
import pickle
from time import perf_counter

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

n_pop = 100
n_gen = 100

optim = FacilityOptimizer(V,
                          n_bays=3,
                          n_pop=n_pop,
                          n_gen=n_gen,
                          c=c,
                          verbose=True)

start = perf_counter()
res = optim.evaluate()
end = perf_counter()

duration = int((end - start)*1000)

with open(f"../data/pickles/facility_optim_pop_{n_pop}_gen_{n_gen}_dur_{duration}ms.pkl", 'wb') as f:
    pickle.dump(res, f)

logger.info("Done")
