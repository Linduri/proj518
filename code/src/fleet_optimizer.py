import pandas as pd
import logging
from compendium import Compendium
import matplotlib.pyplot as plt

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

# Plot locations to graph.
fig, ax = plt.subplots()

ax.scatter(x=c.facs['latitude'],
           y=c.facs['longitude'])

for _, name, lat, lon, _, _, _, _, _ in c.facs.itertuples():
    ax.annotate(str(name).capitalize(),
                (lat,
                 lon),
                xytext=(5, 5),
                textcoords='offset points')

ax.scatter(x=V['latitude'],
           y=V['longitude'])

# Group vehicle IDs
L = V.groupby('loc',
              as_index=False,
              group_keys=False)

for _, l in L:
    names = l['vehicle'].unique()
    names = [str(name) for name in names]
    txt = ','.join(names)
    ax.annotate(txt,
                (l['latitude'].iloc[0],
                 l['longitude'].iloc[0]),
                xytext=(5, -10),
                textcoords='offset points')

# for i, txt in enumerate(V.index):
#     ax.annotate(txt, (V['latitude'].iloc[i],
#                       V['longitude'].iloc[i]))

plt.show()
