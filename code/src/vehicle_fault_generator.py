import pandas as pd
import random
import logging
import datetime
from compendium import Compendium

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

facilities_csv = "../data/facilities.csv"
procedure_names_csv = "../data/procedure_names.csv"
procedure_steps_csv = "../data/procedure_steps.csv"
operations_csv = "../data/operations.csv"
fault_output_csv = "../data/faults.csv"

n_vehicles = 5

start_date = datetime.date(2023, 7, 1)
end_date = datetime.date(2023, 12, 1)

c = Compendium(facilities_csv,
               procedure_names_csv,
               procedure_steps_csv,
               operations_csv)

logger.info(f"Generating {n_vehicles} names...")
ids = random.sample(range(1000, 3000), n_vehicles)
names = [f"VHCL{id}" for id in ids]
logger.info(f"Generated vehicles: {names}")

out_cols = ["vehicle", "procedure", "failure_date"]
faults = pd.DataFrame(columns=out_cols)

logger.info(f"Generating failure data for {n_vehicles} vehicles...")
for idx, name in enumerate(names):
    logger.info(f"Generating vehicle {name}...")

    n_faults = random.randint(1, 5)
    logger.debug((f"Selecting {n_faults} random procedure"
                  f"{'s' if n_faults > 1 else ''}..."))
    vehicle_faults = c.procs.sample(n_faults)
    logger.debug(f"Selected faults: {', '.join(vehicle_faults['name'])}")

    logger.debug(f"Generating {len(vehicle_faults)} failure dates...")
    fault_dates = []
    for _ in vehicle_faults['name']:
        num_days = (end_date - start_date).days
        rand_days = random.randint(1, num_days)
        random_date = start_date + datetime.timedelta(days=rand_days)
        fault_dates.append(random_date)
    fault_dates_strings = [date.strftime('%d/%m/%Y') for date in fault_dates]
    logger.debug(f"Generated dates: {', '.join(fault_dates_strings)}")

    new_rows = pd.DataFrame(
        {"vehicle": [name for _ in range(vehicle_faults.shape[0])],
         "procedure": vehicle_faults["id"],
         "failure_date": fault_dates_strings})

    faults = pd.concat([faults, new_rows], ignore_index=True)

logger.info(faults)
logger.info(f"Saving to csv at {fault_output_csv}")
faults.to_csv(fault_output_csv)
logger.info("Saved!")
