from pymoo.core.callback import Callback


class FacilityCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data["F"] = []
        self.data["F_best"] = []

        self.data["X"] = []
        self.data["x_best"] = []

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        self.data["F"].append(F)
        self.data["F_best"].append(F.min())

        X = algorithm.pop.get("x")
        self.data["X"].append(algorithm.pop.get("x"))
        self.data["x_best"].append(X[F.argmin()])
