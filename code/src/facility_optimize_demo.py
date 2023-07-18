import pandas as pd
import logging
import matplotlib.pyplot as plt
import numpy as np
from compendium import Compendium
from vehicle import Vehicle

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

facilities_csv = "../data/facilities.csv"
procedure_names_csv = "../data/procedure_names.csv"
procedure_steps_csv = "../data/procedure_steps.csv"
operations_csv = "../data/operations.csv"

faults_csv = "../data/faults.csv"

c = Compendium(facilities_csv,
               procedure_names_csv,
               procedure_steps_csv,
               operations_csv)

logger.info("Loading faults...")
faults = pd.read_csv(faults_csv)
logger.info(f"Loaded {len(faults)} faults.")
vehicles = faults['vehicle'].unique()
logger.info(f"Detected {len(vehicles)} vehicles...")
logger.info(vehicles)

V = []
for vehicle in vehicles:
    v = Vehicle(
            vehicle,
            faults.loc[faults["vehicle"] ==
                       vehicle]["procedure"].to_list(),
            faults.loc[faults["failure_date"] ==
                       vehicle]["failure_date"].to_list(),
            c)
    v.optimize()
    V.append(v)

durations = [v.repair_duration() for v in V]
logger.info("Calculated vehicle repair durations...")
logger.info(f"{durations}")

cum_duration = np.cumsum(durations)
logger.info("Calculated cumulative vehicle repair durations...")
logger.info(f"{cum_duration}")

delays = [0]
delays.extend(cum_duration[0:-1])
logger.info("Calculated vehicle repair start delays...")
logger.info(f"{delays}")

fig, axs = plt.subplots(len(vehicles), figsize=(16, 16))
for idx, v in enumerate(V):
    v.optimize()
    series = v.get_ops(delays[idx])
    v.plot_gantt(ax=axs[idx],
                 series=series,
                 x_max=cum_duration[-1])
    axs[idx].set_title(v.name)
