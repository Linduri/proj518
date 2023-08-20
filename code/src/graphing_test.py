import logging
import numpy as np
import pandas as pd
from compendium import Compendium
from problems import Facility
from graphing import PlotBayOps

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

cols = ["v", "p", "i", "b", "c", "o", "s", "d", "oc", "t_e", "t_s"]

B = np.array([[0, 3, 5, 1, 1, 3, 0, 300, 1.0, 300.0, 0],
             [0, 3, 5, 1, 1, 2, 1, 600, 2.0, 900.0, 300],
             [2, 5, 0, 1, 2, 1, 0, 600, 3.0, 1500.0, 900],
             [2, 5, 0, 1, 2, 3, 1, 300, 4.0, 1800.0, 1500],
             [2, 5, 0, 1, 2, 4, 2, 900, 5.0, 2700.0, 1800],
             [2, 5, 0, 1, 2, 6, 3, 600, 6.0, 3300.0, 2700],
             [2, 5, 0, 1, 2, 5, 4, 300, 7.0, 3600.0, 3300],
             [2, 5, 0, 1, 2, 0, 5, 1200, 8.0, 4800.0, 3600],
             [2, 5, 0, 1, 2, 12, 6, 7200, 9.0, 12000.0, 4800],
             [0, 4, 3, 2, 1, 3, 0, 300, 1.0, 300.0, 0],
             [0, 4, 3, 2, 1, 11, 1, 600, 2.0, 900.0, 300],
             [1, 1, 3, 2, 2, 3, 0, 300, 3.0, 1200.0, 900],
             [1, 4, 1, 2, 2, 3, 0, 300, 3.0, 1200.0, 900],
             [1, 1, 3, 2, 2, 4, 1, 900, 4.0, 2400.0, 1500],
             [1, 4, 1, 2, 2, 11, 1, 600, 5.0, 3000.0, 2400],
             [1, 1, 3, 2, 2, 6, 2, 600, 6.0, 3600.0, 3000],
             [1, 1, 3, 2, 2, 9, 3, 1200, 7.0, 4800.0, 3600],
             [1, 2, 2, 3, 1, 3, 0, 300, 1.0, 300.0, 0],
             [1, 2, 2, 3, 1, 2, 1, 600, 2.0, 900.0, 300],
             [1, 2, 2, 3, 1, 10, 2, 180, 3.0, 1080.0, 900]])

D = pd.DataFrame(columns=cols,
                 data=B)

PlotBayOps(D)
