from typing import List
import pandas as pd
import numpy as np
from vehicle import Vehicle


class Facility:

    def __init__(self,
                 name: str,
                 latitude: float,
                 longitude: float,
                 start: int,
                 open: int,
                 close: int,
                 stop: int,
                 bays: int):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.start = start
        self.open = open
        self.close = close
        self.stop = stop
        self.bays = bays

    def info(self):
        return (f"{self.name}"
                f" LOC[{self.latitude},"
                f" {self.longitude}]"
                f" ONLINE[{self.start} - {self.stop}]"
                f" OPEN [{self.open} - {self.close}]"
                f" BAYS [{self.bays}]")

    # Optimizer data structure
    #
    # row | vehicle | procedure | bay
    #  X  |    Y    |     Z     |  W

    def optimize(self, V: List[Vehicle], P: List[tuple]):
        """
        Find the optimal schedule for a collection of
        vehicles and selected procedures.

        Parameters:
        V (List[int]): List of vehicles.

        P (List[tuple]): List of procedures per vehicle.
        tuple(vehicle, procedure).

        """

        # Convert input data to optimizer table.
        D = pd.DataFrame(
            columns=['v', 'p'],
            data=P
        )

        self.evaluate(D)

    def evaluate(self, D):
        # v = vehicle
        # p = procedure
        # b = bay
        # o = order
        # P = procedure group

        # Assign procedures to bays.
        D['b'] = np.random.randint(1, self.bays+1, len(D))

        # Assign bay order
        D['o'] = D.groupby('b').cumcount()

        # Sort by bay and order
        D.sort_values(by=['b', 'o'], inplace=True)

        # Group adjacent vehicle procedures (P).
        D['P'] = D.groupby(
                    'b',
                    as_index=False,
                    group_keys=False
                ).apply(
                    lambda b: (b['v'] != b.v.shift()).cumsum()
                )

        # Unpack procedure groups into optimized operations.
        
        
        print(D)

        # Optimize adjacent procedures to same vehicle.
        # Evaluate solution
        # Iterate
