import pandas as pd
import logging
from compendium import Compendium
from graphing import PlotVehicleLocations

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

PlotVehicleLocations(V[['vehicle', 'loc', 'latitude', 'longitude']],
                     c.facs[['name', 'latitude', 'longitude']])
