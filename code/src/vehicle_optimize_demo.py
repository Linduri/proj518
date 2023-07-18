import pandas as pd
import logging
from compendium import Compendium
from vehicle import Vehicle

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

facilities_csv = "../data/facilities.csv"
procedure_names_csv = "../data/procedure_names.csv"
procedure_steps_csv = "../data/procedure_steps.csv"
operations_csv = "../data/operations.csv"

vehicle_faults_input_csv = "../data/vehicle_faults.csv"

c = Compendium(facilities_csv,
               procedure_names_csv,
               procedure_steps_csv,
               operations_csv)

logger.info("Loading faults...")
faults = pd.read_csv(vehicle_faults_input_csv)
logger.info(f"Loaded {len(faults)} faults.")

v = Vehicle(faults["vehicle"][0],
            faults["procedure"].to_list(),
            faults["failure_date"].to_list(),
            c)

v.plot_gantt()
v.optimize_repairs()
v.plot_gantt()
