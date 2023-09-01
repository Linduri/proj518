import numpy as np
from facility_problem import Facility
from mutators import BayMutator
import pandas as pd
import logging
from compendium import Compendium

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

facilities_csv = "../data/facilities.csv"
procedure_names_csv = "../data/procedure_names.csv"
procedure_steps_csv = "../data/procedure_steps.csv"
operations_csv = "../data/operations.csv"

c = Compendium(facilities_csv,
               procedure_names_csv,
               procedure_steps_csv,
               operations_csv)

V0 = np.array([[0, 3, 0, 1],
              [0, 4, 1, 3],
              [1, 1, 2, 3],
              [1, 2, 3, 3],
              [1, 4, 4, 2],
              [2, 5, 5, 3]])

V1 = np.array([[0, 3, 5, 1],
              [0, 4, 4, 2],
              [1, 1, 3, 3],
              [1, 2, 2, 3],
              [1, 4, 1, 2],
              [2, 5, 0, 1]])

V = np.vstack((V0.flatten(),
              V1.flatten()))

D = pd.DataFrame(columns=['v', 'p', 'i', 'b'],
                 data=V0)

# Unpack procedures to get operations and steps.
#  __________________________________________
# | procedure | operation | step  | duration |
# |    (p)    |    (o)    |  (s)  |    (d)   |
# |===========|===========|=======|==========|
# |    int    |    int    |  int  |    int   |
# |    ...    |    ...    |  ...  |    ...   |
logger.info("Unpacking procedures...")
P = D.p.unique()
od_dict = c.ops.set_index('id').to_dict()['duration']
ops = pd.DataFrame(columns=['p', 'o', 's', 'd'])
for p in P:
    s = c.steps[c.steps.procedure == p].copy()
    s['duration'] = s.operation.replace(od_dict)
    s.rename(columns={'procedure': 'p',
                      'operation': 'o',
                      'step': 's',
                      'duration': 'd'},
             inplace=True)

    ops = pd.concat([ops, s],
                    ignore_index=True)

logger.info("Unpacked procedures.")

n_var = 2
n_bays = 3

p = Facility(n_bays=n_bays,
             n_cols=4,
             ops=ops)
m = BayMutator(n_pop=len(V),
               prob=0.5)

logger.info(f"\n{V.reshape(len(V), -1, 4)}")
Vm = m._do(problem=p, X=V)
logger.info("Mutated")
logger.info(f"\n{Vm.reshape(len(V), -1, 4)}")
