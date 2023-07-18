import pandas as pd
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
from compendium import Compendium


class Vehicle:

    def __init__(self,
                 name,
                 faults,
                 failure_dates,
                 c: Compendium):
        self.name = name
        self.faults = faults
        self.dates = failure_dates
        self.c = c

        self.logger = logging.getLogger()
        self.logger.info("Unpacking operations...")
        out_cols = ["procedure", "operation", "step"]
        self.steps = pd.DataFrame(columns=out_cols)
        for fault in faults:
            fault_steps = c.steps.loc[c.steps["procedure"] == fault]
            n_steps = fault_steps.shape[0]

            duration_dict = c.ops["duration"].to_dict()
            durations = fault_steps["operation"].map(duration_dict)

            _rows = pd.DataFrame({"procedure": [fault for _ in range(n_steps)],
                                  "operation": fault_steps["operation"],
                                  "step": fault_steps["step"],
                                  "duration": durations})

            self.steps = pd.concat([self.steps, _rows], ignore_index=True)

        self.logger.info(f"Unpacked {len(self.steps)} operations "
                         f"for {len(faults)} procedures.")

        n_duplicates = len(self.steps["operation"]) - \
            len(self.steps["operation"].drop_duplicates())

        self.logger.info(f"Detected {n_duplicates} duplicate operations.")

        self.logger.debug("Grouping similar operations...")
        self.series = self.steps.sort_values(["procedure", "step"])
        self.series.reset_index(inplace=True)

        self.logger.debug("Calculating start times...")
        self.series["start"] = \
            self.series.duration.cumsum() - self.series.duration

        self.series = self.series.merge(c.ops, left_on='operation',
                                        right_on="id")
        self.series = self.series.drop("duration_y", axis=1)
        self.series.rename(columns={'duration_x': 'duration'}, inplace=True)
        self.series = self.series.sort_values(["start"])

        self.logger.debug("Generating procedure group colors...")
        n_faults = len(faults)
        cmap = plt.get_cmap("gist_rainbow", n_faults)
        custom_palette = [mpl.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
        unique = pd.DataFrame({"procedure": self.series["procedure"].unique(),
                               "color": custom_palette})
        self.series = self.series.merge(unique,
                                        left_on="procedure",
                                        right_on="procedure")

        self.series.name = self.series.name + " " + \
            self.series.procedure.astype(str)

    def optimize(self):
        self.logger.info((f"Optimizing {len(self.series)} "
                         "operation scheduling..."))
        X = self.series.operation.unique()
        for x in X:
            Y = self.series.loc[self.series.operation == x]

            if len(Y) > 1:
                i = 1
                for idx, y in Y.iloc[1:].iterrows():
                    i = Y.index[0] + 0.0001*i
                    self.series.loc[i] = y
                    self.series.drop(idx, inplace=True)
                    i += 1

                self.series.sort_index(inplace=True)
                self.series.reset_index(drop=True, inplace=True)

        self.logger.debug("Calculating parallel start times...")
        self.series.iloc[0, self.series.columns.get_loc('start')] = 0
        for idx, row in self.series[1:].iterrows():
            self.series.iloc[idx, self.series.columns.get_loc('start')] = \
                self.series.iloc[idx-1].start
            if self.series.iloc[idx-1].operation != row.operation:
                self.series.iloc[idx, self.series.columns.get_loc('start')] +=\
                    self.series.iloc[idx-1].duration

        self.logger.info(f"Optimized {len(self.series)} operation schedule.")

    def repair_duration(self):
        return self.series.tail(1).start.values[0] + \
            self.series.tail(1).duration.values[0]

    def plot_gantt(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(16, 6))

        ax.barh(self.series.name,
                self.series.duration,
                left=self.series.start,
                color=self.series.color)
        ax.set_xlim(0,
                    self.repair_duration())

        if ax is None:
            plt.show()
