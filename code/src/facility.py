from typing import List
import pandas as pd
from compendium import Compendium
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

    def print_info(self):
        print((f"{self.name}"
               f" LOC[{self.latitude},"
               f" {self.longitude}]"
               f" ONLINE[{self.start} - {self.stop}]"
               f" OPEN [{self.open} - {self.close}]"
               f" BAYS [{self.bays}]"))

    # Optimizer data structure
    #
    # row | vehicle | procedure | bay
    #  X  |    Y    |     Z     |  W

    def optimize(V: List[Vehicle], P: List[tuple]):
        # Convert input data to optimizer table.
        D = pd.DataFrame(
            columns=['v', 'p'],
            data=P
        )

        print(D)

        # Assign procedures to bays.

        # Optimize adjacent procedures to same vehicle.
        # Evaluate solution
        # Iterate
