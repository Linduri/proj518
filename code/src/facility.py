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

    def rand_bays(self, n: int):
        """
        Generates n random bay ids between 1 
        (inclusive) and self.bays + 1 (exclusive).

        Parameters:
        n (int): Number of bay ids to generate.

        Returns:
        List[int]: List of bay ids.

        """

        return np.random.randint(1, self.bays+1, n)

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

        # Assign procedures to bays.
        D['b'] = self.rand_bays(len(D))
        
        print(D)
        
        # Optimize adjacent procedures to same vehicle.
        # Evaluate solution
        # Iterate
