from pymoo.core.callback import Callback
import matplotlib.pyplot as plt


class OptimizerCallback(Callback):
    """Records data from each optimiser generation."""

    def __init__(self) -> None:
        super().__init__()
        self.data["F"] = []
        self.data["F_opt"] = []

        self.data["X"] = []
        self.data["x_best"] = []
        self.data["x_f_min"] = []

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        self.data["F"].append(F)

        F_opt = algorithm.opt.get("F")
        self.data["F_opt"].append(F_opt[0])

        X = algorithm.opt.get("x")
        self.data["x_best"].append(X)


class ObjectiveSpaceAnimation(Callback):
    """Prints the pareto front for each optimiser generation.
    """

    def _update(self, algorithm):

        if algorithm.n_gen % 5 == 0:
            F = algorithm.opt.get("F")
            pf = algorithm.problem.pareto_front()

            plt.clf()
            plt.scatter(F[:, 0], F[:, 1])
            if pf is not None:
                plt.plot(pf[:, 0], pf[:, 1], color="black", alpha=0.7)

            plt.show()
