import pandas as pd
import logging


class Compendium:
    """Store static data for easy management.
    """

    def __init__(self,
                 facilities_path,
                 procedure_names_path,
                 procedure_steps_path,
                 operations_path):

        self.logger = logging.getLogger(__name__)

        self.logger.info("Loading compendium...")
        self.logger.info("Loading facilities...")
        self.facs = pd.read_csv(facilities_path)
        self.logger.info(f"Loaded {len(self.facs)} facilities.")

        self.logger.info("Loading procedure names...")
        self.procs = pd.read_csv(procedure_names_path)
        self.logger.info((f"Loaded {len(self.procs)} unique "
                         "procedures."))

        self.logger.info("Loading procedures steps...")
        self.steps = pd.read_csv(procedure_steps_path)
        self.logger.info((f"Loaded {len(self.steps)} procedure "
                         "steps."))

        self.logger.info("Loading operations...")
        self.ops = pd.read_csv(operations_path)
        self.logger.info(f"Loaded {len(self.ops)} operations.")
        self.logger.info("Loaded compendium!")
