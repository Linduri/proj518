from typing import List
import pandas as pd
from compendium import Compendium


class Vehicle:

    def __init__(self,
                 id: str,
                 procedures: List[int],
                 procedure_dates,
                 c: Compendium):
        self.id = id
        self.c = c

        self.F = pd.DataFrame(
            {'p': procedures,
             't': procedure_dates}
        )

        self.ops = c.steps.loc[
            c.steps.procedure.isin(self.F.p.unique())
            ]

    def P(self) -> List[int]:
        """Gets the needed procedures for this vehicle.

        Returns:
            List[int]: List of procedure IDs.
        """
        return self.F.p.to_list()

    def T(self) -> List[int]:
        """Gets the datetimes each procedure needs to be done by.

        Returns:
            List[int]: List of datetimes.
        """
        return self.F.t.to_list()
    
    def get_ops(self, P: List[int]):
        """
        Get an optimized sequence of operations for
        procedures P.

        Args:
            P (List[int]): List of procedure IDs to get.
        """
