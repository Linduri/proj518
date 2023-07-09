import pandas as pd
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

procedure_names_csv = "../data/procedure_names.csv"
steps_csv = "../data/procedure_steps.csv"
operations_csv = "../data/operations.csv"

vehicle_faults_input_csv = "../data/vehicle_faults.csv"

logger.info("Loading procedure names...")
procedure_names = pd.read_csv(procedure_names_csv)
logger.info(f"Loaded {len(procedure_names)} unique procedures.")

logger.info("Loading procedures steps...")
steps = pd.read_csv(steps_csv)
logger.info(f"Loaded {len(steps)} procedure steps.")

logger.info("Loading operations...")
operations = pd.read_csv(operations_csv)
logger.info(f"Loaded {len(operations)} operations.")

logger.info("Loading faults...")
faults = pd.read_csv(vehicle_faults_input_csv)
logger.info(f"Loaded {len(faults)} faults.")

logger.info("Unpacking operations...")
out_cols = ["procedure", "operation", "step"]
all_steps = pd.DataFrame(columns=out_cols)
for idx, row in faults.iterrows():
    i_fault = row["procedure"]
    fault_steps = steps.loc[steps["procedure"] == i_fault]
    n_steps = fault_steps.shape[0]

    duration_dict = operations["duration"].to_dict()
    durations = fault_steps["operation"].map(duration_dict)

    new_rows = pd.DataFrame({"procedure": [i_fault for _ in range(n_steps)],
                             "operation": fault_steps["operation"],
                             "step": fault_steps["step"],
                             "duration": durations})

    all_steps = pd.concat([all_steps, new_rows], ignore_index=True)

logger.info(f"Unpacked {len(all_steps)} operations "
            f"for {len(faults)} procedures.")

all_steps = all_steps.sort_values(["operation", "step"])
print(all_steps)

# max_steps = all_steps.step.max()
# for idx, fault in faults.iterrows():
#     X = []
#     i_procedure = fault["procedure"]
#     X.append(i_procedure)
#     X.append(all_steps.loc[all_steps["procedure"] == i_procedure]["operation"])
#     print(X)
    
# df = pd.DataFrame([['A', 10, 20, 10, 30], ['B', 20, 25, 15, 25], ['C', 12, 15, 19, 6],
#                    ['D', 10, 29, 13, 19]],
#                   columns=['Team', 'Round 1', 'Round 2', 'Round 3', 'Round 4'])

# logger.info("Generating Gantt...")
# out_cols = ["task", "start", "duration", "group"]
# gantt = pd.DataFrame(columns=out_cols)
# for idx, step in all_steps.iterrows():
#     operation = operations.loc[operations["id"] == step["operation"]]
#     new_row = pd.DataFrame(
#         {"task": f"({step['procedure']}) " + operation["name"],
#          "start": 0,
#          "duration": operation["duration"],
#          "group": step['procedure']}
#         )
#     gantt = pd.concat([gantt, new_row], ignore_index=True)

# fig, ax = plt.subplots(1, figsize=(16, 6))
# ax.barh(gantt.task, gantt.duration, left=gantt.start)
# plt.show()
