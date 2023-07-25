import pandas as pd
import logging
from compendium import Compendium
from vehicle import Vehicle

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

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
F = pd.read_csv(vehicle_faults_input_csv)
logger.info(f"Loaded {len(F)} faults.")


V = [Vehicle(v,
             F.loc[F.vehicle == v].procedure,
             F.loc[F.vehicle == v].failure_date,
             c) for v in F.vehicle.unique()
     ]

logger.debug(f"Loaded {len(V)} vehicle{'s' if len(V) > 1 else ''}.")

for v in V:
    logger.info(f"{v.id} has procedures {v.P()}")

logger.info("Vehicle 0 procedures have ops...")
logger.info(V[1].get_ops(V[1].P()))
