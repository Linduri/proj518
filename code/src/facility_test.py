import pandas as pd
import logging
from compendium import Compendium
from vehicle import Vehicle
from facility import Facility

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
VF = pd.read_csv(vehicle_faults_input_csv)
logger.info(f"Loaded {len(VF)} faults.")


V = [Vehicle(v,
             VF.loc[VF.vehicle == v].procedure,
             VF.loc[VF.vehicle == v].failure_date,
             c) for v in VF.vehicle.unique()
     ]

logger.debug(f"Loaded {len(V)} vehicle{'s' if len(V) > 1 else ''}.")

F = [Facility(f.name,
              f.latitude,
              f.longitude,
              f.start,
              f.open,
              f.close,
              f.stop,
              f.bays) for idx, f in c.facs.iterrows()
     ]

for f in F:
    f.print_info()
