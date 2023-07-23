from typing import List
import pandas as pd
from compendium import Compendium


class Vehicle:

    def __init__(self,
                 name: str,
                 procedures: List[int],
                 procedure_dates,
                 c: Compendium):
        self.name = name
        self.c = c

        self.F = pd.DataFrame(
            {'p': procedures,
             't': procedure_dates}
        )

        self.ops = c.steps.loc[
            c.steps.procedure.isin(self.F.p.unique())
            ]

    # Return list of procedure IDs for this vehicle.
    def P(self) -> List[int]:
        return self.F.p.to_list()

    # Return list of dates each procedure needs to be done by.
    def T(self) -> List[int]:
        return self.F.t.to_list()
