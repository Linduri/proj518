import sys
import numpy as np
from pymoo.core.problem import ElementwiseProblem


class Facility(ElementwiseProblem):

    def __init__(self,
                 n_var,
                 n_bays,
                 **kwargs):
        
        self.n_var = n_var
        
        super().__init__(n_var=self.n_var,
                         n_obj=1,
                         xl=[0, 0],
                         xu=[sys.maxsize, n_bays],
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluates the facility problem.

        Args:
            x (_type_): Population member to evaluate.
             _____________________________________________
            |  vehicle_procedure_id  |  priority  |  bay  |
            |========================|============|=======|
            |          int           |    int     |  int  |

            out (_type_): Evaluation results.
        """
        x = np.reshape(x, (-1, self.n_var+1))
        print(x)

        out['F'] = np.array([1])
