from pymoo.core.crossover import Crossover


class BayCrossover(Crossover):

    def __init__(self, shift=False, prob=0.5, **kwargs):
        n_parents = 2
        n_offspring = 2
        super().__init__(n_parents,
                         n_offspring,
                         **kwargs)
        self.shift = shift

    def _do(self, problem, X, **kwargs):
        # Reshape to four columns...
        #  ______________________________________
        # | vehicle | procedure | priority | bay |
        # |=========|===========|==========|=====|
        # |   int   |    int    |    int   | int |
        # |   ...   |    ...    |    ...   | ... |
        _X = X.copy().reshape((-1, 4))

        

        return _X.reshape(X.shape)
