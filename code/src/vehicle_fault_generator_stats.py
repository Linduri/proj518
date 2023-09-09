import pandas as pd
import numpy as np
import random
import logging
import datetime
from compendium import Compendium
from time import perf_counter
from statistics import mean
import matplotlib.pyplot as plt

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

facilities_csv = "../data/facilities.csv"
vehicle_locations_csv = "../data/vehicle_locations.csv"
procedure_names_csv = "../data/procedure_names.csv"
procedure_steps_csv = "../data/procedure_steps.csv"
operations_csv = "../data/operations.csv"
fault_output_csv = "../data/vehicle_faults.csv"

n_vehicles = 5

start_date = datetime.date(2023, 7, 1)
end_date = datetime.date(2023, 12, 1)

c = Compendium(facilities_csv,
               procedure_names_csv,
               procedure_steps_csv,
               operations_csv)

res = []
n_range = np.array([*range(0, 105, 5)])
for n_vehicles in n_range:
    res_trial = []
    logger.info(f"N vehicles {n_vehicles}")
    for trial in range(100):
        logger.info(f"Trial {trial}")
        start_time = perf_counter()

        logger.debug(f"Generating {n_vehicles} names...")
        ids = random.sample(range(1000, 3000), n_vehicles)
        names = [f"VHCL{id}" for id in ids]
        logger.debug(f"Generated vehicles: {names}")

        out_cols = ["vehicle", "procedure", "failure_date"]
        faults = pd.DataFrame(columns=out_cols)

        logger.debug(f"Generating failure data for {n_vehicles} vehicles...")
        for idx, name in enumerate(names):
            logger.debug(f"Generating vehicle {name}...")

            n_faults = random.randint(1, 5)
            logger.debug((f"Selecting {n_faults} random procedure"
                         "{'s' if n_faults > 1 else ''}..."))
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
            fault_dates_epochs = [date.strftime('%s') for date in fault_dates]
            logger.debug(f"Generated dates: {', '.join(fault_dates_strings)}")

            new_rows = pd.DataFrame(
                {"vehicle": [idx for _ in range(vehicle_faults.shape[0])],
                 "procedure": vehicle_faults["id"],
                 "failure_date": fault_dates_epochs})

            faults = pd.concat([faults, new_rows], ignore_index=True)

        logger.debug("Generating vehicle locations...")
        locs = pd.read_csv(vehicle_locations_csv)
        # Randomize locations
        locs = locs.sample(frac=1)

        rng = np.random.default_rng()
        i_max = n_vehicles if n_vehicles < len(locs) else len(locs)
        i_rnd = rng.integers(0,
                             i_max,
                             n_vehicles)

        i_arr = np.arange(n_vehicles)

        loc_dict = (dict(zip(i_arr, i_rnd)))

        lat_dict = (dict(zip(i_arr, locs['latitude'])))
        lon_dict = (dict(zip(i_arr, locs['longitude'])))

        faults['loc'] = faults["vehicle"].map(loc_dict)

        faults["latitude"] = faults["loc"].map(lat_dict)
        faults["longitude"] = faults["loc"].map(lon_dict)

        logger.debug("Generated vehicle locations.")

        logger.debug(faults)
        logger.debug(f"Saving to csv at {fault_output_csv}")
        faults.to_csv(fault_output_csv,
                      index=False)
        logger.debug("Saved!")

        end_time = perf_counter()
        duration = (end_time - start_time)*1000
        res_trial.append(duration)

    res.append(res_trial)

arr = np.array(res)
n_vals = len(arr.flatten())

res_trial = np.empty((0, 2))
res_mean = np.empty((len(arr), 1))
res_lwr = np.empty((len(arr), 1))
res_upr = np.empty((len(arr), 1))
for i, trial in enumerate(arr):
    res_mean[i] = mean(trial)
    res_std = np.std(trial)
    res_lwr[i] = res_mean[i] - res_std
    res_upr[i] = res_mean[i] + res_std

    T = np.empty((len(trial), 2))
    T[:, 0] = [n_range[i] for _ in range(len(T[:, 0]))]
    T[:, 1] = trial.T
    res_trial = np.concatenate([res_trial, T])

fig, axs = plt.subplots(1,
                        1,
                        sharex=True)

axs.plot(n_range,
         res_mean,
         linestyle="dashed",
         color="black")
axs.plot(n_range,
         res_lwr,
         linestyle=":",
         color="black")
axs.plot(n_range,
         res_upr,
         linestyle=":",
         color="black")

axs.scatter(res_trial[:, 0],
            res_trial[:, 1],
            s=0.25,
            color="black")

axs.set_ylabel("Duration (ms)")

plt.xlabel("Vehicles")
# plt.xticks(range(len(n_range)), n_range)
# plt.ylabel("F")
plt.show()
