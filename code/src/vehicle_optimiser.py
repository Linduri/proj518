import pandas as pd
import logging

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
    new_rows = pd.DataFrame({"procedure": [i_fault for _ in range(n_steps)],
                             "operation": fault_steps["operation"],
                             "step": fault_steps["step"]})

    all_steps = pd.concat([all_steps, new_rows], ignore_index=True)

logger.info(f"Unpacked {len(all_steps)} operations"
            f"for {len(faults)} procedures.")


