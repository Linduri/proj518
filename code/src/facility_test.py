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

VD = pd.DataFrame({
    'v': [v.id for v in V for _ in v.P()],
    'p': [p for v in V for p in v.P()]
})

F = [Facility(f.name,
              f.latitude,
              f.longitude,
              f.start,
              f.open,
              f.close,
              f.stop,
              f.bays) for _, f in c.facs.iterrows()
     ]

logger.debug(f"Loaded {len(F)} facilit{'ies' if len(F) > 1 else 'y'}.")

for f in F:
    logger.debug(f.info())

logger.info("Randomly sampling 60% of faults...")
VDs = VD.sample(frac=0.6)

F[0].optimize(
    V,
    list(VDs.itertuples(index=False, name=None))
)
