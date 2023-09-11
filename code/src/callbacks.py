from pymoo.core.callback import Callback
import matplotlib.pyplot as plt


class OptimizerCallback(Callback):
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
        # self.data["F_best"].append(F.min())

        # X = algorithm.pop.get("x")
        # self.data["x_f_min"].append(X[F.argmin()])
        # self.data["X"].append(algorithm.pop.get("x"))
        # G = algorithm.pop.get("G")
        # df = pd.DataFrame(columns=['X', 'F', 'G'])
        # df['X'] = np.arange(len(X))
        # df['F'] = F
        # df['G'] = G

        # df = df.loc[df['G'] < 0]
        # df = df.sort_values(by=['F'],
        #                     ascending=True)
        # best = df.head(1)
        # best = None if best.empty else X[best['X']]
        # self.data["x_best"].append(best)


class ObjectiveSpaceAnimation(Callback):

    def _update(self, algorithm):

        if algorithm.n_gen % 5 == 0:
            F = algorithm.opt.get("F")
            pf = algorithm.problem.pareto_front()

            plt.clf()
            plt.scatter(F[:, 0], F[:, 1])
            if pf is not None:
                plt.plot(pf[:, 0], pf[:, 1], color="black", alpha=0.7)

            plt.show()
