import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from planner import Mission, Strategy

pwd = os.getcwd()
import colorsys


class OLD:

    def __init__(self):
        # self.csv_name = csv_name
        # self.output_graph_path = os.path.join(path, self.csv_name + ".png")
        # self.csv_path = os.path.join(path, self.csv_name)
        self.data = self.load_csv_to_numpy()
        self.names = 0
        self.times = 1
        self.errors = 2
        self.streaks = 3
        self.integral = 4
        self.integral_nn = 5
        self.data_counts = 6
        self.data_counts_filtered = 7
        self.gt_type = 8
        self.t = self.get_N_HexCol(len(self.data))
        # self.sort =  [x[-1] for x in self.data[:, self.names]]
        self.data = self.data[np.array(self.data[:, self.data_counts_filtered], dtype=int).argsort()]
        self.percentages = np.array([x[0:4] for x in self.data[:, self.names]]).astype(float)
        self.block_size = np.array([x[5] for x in self.data[:, self.names]]).astype(float)
        self.whole_place = np.array([x[7] for x in self.data[:, self.names]]).astype(float)
        self.sinlge_per_block = np.array([x[9] for x in self.data[:, self.names]]).astype(float)
        self.cestlice = np.array([x[11:14] for x in self.data[:, self.names]]).astype(float)
        self.strands = np.array([x[14:17] for x in self.data[:, self.names]]).astype(float)
        self.data_count_arr = np.array(self.data[:, self.data_counts_filtered], dtype=int)
        self.data_count_u_arr = np.array(self.data[:, self.data_counts], dtype=int)
        self.integral_arr = np.array(self.data[:, self.integral], dtype=float)
        self.integral_nn_arr = np.array(self.data[:, self.integral_nn], dtype=float)
        self.gt_type_arr = np.array(self.data[:, self.gt_type], dtype=str)
        # load freom pickle
        self.data_frames = []
        paths = []
        # for i, data in enumerate(self.data):
        # p = os.path.join(path, self.data[i][self.names], "usage.pickle")
        # paths.append(p)
        # d = pickle.load(open(p, "rb"))
        # if p in paths:
        #    self.data_frames.append(d)


def get_only_keys(keys, dictionary):
    out = {}
    for key in dictionary:
        if key in keys:
            out[key] = dictionary[key]
    return out


class Results:
    def __init__(self, path):
        self.path = path
        self.missions = self.load_missions(path)
        self.output_graph_path = os.path.join(path, "compare.png")
        # self.t = self.get_N_HexCol(len(self.data))

    def load_missions(self, path):
        experiments_path = os.path.join(pwd, path)
        paths = [item for item in os.listdir(experiments_path) if os.path.isdir(os.path.join(experiments_path, item))]
        paths.sort()
        missions = []
        for mission_name in paths:
            mission = Mission(int(mission_name))
            mission = mission.load(os.path.join(path, mission_name))
            missions.append(mission)
        return missions

    def get_N_HexCol(self, N=5):
        HSV_tuples = [(x * 1.0 / (N * 1.5), 1, 1) for x in range(N)]
        hex_out = []
        for rgb in HSV_tuples:
            rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
            hex_out.append('#%02x%02x%02x' % tuple(rgb))
        return hex_out

    def is_strategy_same_as_params(self, strategy, params, exclude):

        strategy_keys = ["uptime", "block_size", "dataset_weights", "used_teach_count", "place_weights", "time_limit",
                         "time_advance", "change_rate", "iteration", "duty_cycle"]
        p = get_only_keys(strategy_keys, vars(params))
        m = get_only_keys(strategy_keys, vars(strategy))
        e = get_only_keys(strategy_keys, vars(exclude))
        # compare dicts p and m and return true if non None values from p are same in m
        for key in p:
            if p[key] is not None:
                if key in ["used_teach_count"]:
                    if p[key] == 0:
                        continue
                    if p[key] < m[key]:
                        return False
                elif key in ["place_weights"]:
                    if not np.array_equal(p[key], m[key]):
                        return False
                elif p[key] != m[key]:
                    return False

        for key in e:
            if e[key] is not None:
                if key in ["used_teach_count"]:
                    if e[key] == 0:
                        continue
                    if e[key] >= m[key]:
                        return False
                elif key in ["place_weights"]:
                    if np.array_equal(p[key], m[key]):
                        return False
                elif e[key] == m[key]:
                    return False

        return True

    def filter_strategies(self, mission_params=None, stategy_params=None, exclude_strategy=None, sorting_params=None):

        # dam inclusion param pro mise // napr mise se specifickym polem na startu
        # dam inclusion param pro strategie //napr pouze 3. iterace
        # dam sorting param
        # ono to vyradi a sortne v poradi strategie
        values = []
        strategies = []
        for mission in self.missions:
            for strategy in mission.old_strategies:
                if not self.is_strategy_same_as_params(strategy, stategy_params, exclude_strategy):
                    continue
                if getattr(strategy, sorting_params) not in values:
                    values.append(getattr(strategy, sorting_params))
                strategies.append(strategy)

        # TODO make this by sortin_params, prolly use getattribute....
        strategies.sort(key=lambda x: getattr(x, sorting_params), reverse=False)
        colors = self.get_N_HexCol(len(strategies))
        return strategies, colors, values

    def scatter(self, filter_strategy=Strategy(), sorting_paramteres="change_rate", exclude_strategy=Strategy()):
        fig = plt.figure()
        ax = plt.gca()
        strategies_to_plot, colors, values = self.filter_strategies(stategy_params=filter_strategy,
                                                                    sorting_params=sorting_paramteres,
                                                                    exclude_strategy=exclude_strategy)
        GT_versions = ["strands", "grief"]
        for s, strategy in enumerate(strategies_to_plot):
            grading = strategy.grading
            for grade in grading:
                if grade is None:
                    continue
                if grade.name == "grief":
                    continue
                style = "."
                styles = [".", "P", "*", "d"]
                t = strategy.title_parameters()

                if len(values) <= len(styles):
                    style = styles[values.index(getattr(strategy, sorting_paramteres))]
                ax.scatter(getattr(strategy, sorting_paramteres), grade.AC_fm_integral,
                           label=t, c=colors[s],
                           alpha=0.5, marker=style)
                ax.annotate(t, (getattr(strategy, sorting_paramteres), grade.AC_fm_integral))

        ax.set_xscale('symlog')
        plt.ylabel("Integral")
        plt.xlabel(sorting_paramteres)
        # plt.legend()
        # plt.savefig(self.output_graph_path)

    def plot(self, filter_strategy=Strategy(), sorting_paramteres="change_rate",exclude_strategy=Strategy()):
        fig = plt.figure()
        ax = plt.gca()
        strategies_to_plot, colors, values = self.filter_strategies(stategy_params=filter_strategy,
                                                                    sorting_params=sorting_paramteres,
                                                                    exclude_strategy=exclude_strategy)
        GT_versions = ["strands", "grief"]
        for s, strategy in enumerate(strategies_to_plot):
            grading = strategy.grading
            for grade in grading:
                if grade is None:
                    continue
                if grade.name == "grief":
                    pass
                    continue
                style = "solid"
                styles = ["solid", "dashed", "dashdot", "dotted"]
                if len(values) <= len(styles):
                    style = styles[values.index(getattr(strategy, sorting_paramteres))]
                    print(values.index(getattr(strategy, sorting_paramteres)))
                line_fm = grade.AC_fm
                line_nn = grade.AC_nn
                ax.plot(line_fm[0], line_fm[1], label=strategy.title_parameters(), c=colors[s], alpha=0.5,
                        linestyle=style)
                # ax.plot(line_nn[0], line_nn[1], label=strategy.title_parameters(), c=colors[s], alpha=0.5,
                #        linestyle=style)

        plt.axvline(x=0.035, color='k')
        plt.legend()
        plt.xlim([0, 0.5])
        plt.title(filter_strategy.title_parameters())
        plt.xlabel("allowed image shift")
        plt.ylabel("probability of correct registration")

    def get_corr_from_strategy(self, startegies, var=[], type="standart"):
        out = []
        text = []
        for v in var:
            o = []
            t = []
            for s in startegies:
                if type == "grade":
                    for grade in s.grading:
                        if grade is None:
                            continue
                        if grade.name == "grief":
                            continue
                        o.append(getattr(grade, v))
                        t.append(str(v))
                else:
                    o.append(getattr(s, v))
                    t.append(str(v))
            out.append(o)
            text.append(t)
        return out, text

    def correlate(self, filter_strategy=Strategy(), sorting_parameters="change_rate", correlation_var=[],
                  grading_var=["AC_fm_integral"], exclude_strategy=Strategy()):
        strategies_to_corelate, colors, values = self.filter_strategies(stategy_params=filter_strategy,
                                                                        sorting_params=sorting_parameters,
                                                                        exclude_strategy=exclude_strategy)

        v1, t1 = (self.get_corr_from_strategy(strategies_to_corelate, correlation_var))
        v2, t2 = (self.get_corr_from_strategy(strategies_to_corelate, grading_var, type="grade"))
        corr = np.concatenate([v1, v2])
        names = np.concatenate([correlation_var, grading_var])

        R2 = np.corrcoef(corr)
        df_cm = pd.DataFrame(R2, index=[i for i in names],
                             columns=[i for i in names])
        plt.figure()
        sn.heatmap(df_cm, annot=True)

    def plot_recognition_corelation(self):
        a = []
        b = []
        cof = []
        names = []
        for frame in self.data_frames:
            aa = frame.loc[
                "P(used position)"].values  # frame.loc["P(used position)"].values[0] + frame.loc["P(used position)"].values[3])
            name = frame.loc["P(used position)"].values[0] + frame.loc["P(used position)"].values[3]
            bb = frame.loc["used bad"].values
            names.append(name)
            a.append(aa)
            b.append(bb)
            cof.append(np.corrcoef(aa, bb)[0, 1])
        # df = pd.DataFrame(data=[a,b])
        # df_cm = pd.DataFrame(R2, index=[i for i in names],
        #                     columns=[i for i in names])
        plt.figure()
        R2 = np.corrcoef(np.array(a).flatten(), np.array(b).flatten())

        print(R2)
        sn.heatmap(R2)
        R2 = np.corrcoef(np.array(a), np.array(b))

        plt.figure()
        # n.heatmap(R2)
        print(cof)
        plt.scatter(names, cof)  # label=names)
        plt.legend()


if __name__ == "__main__":
    results = Results(os.path.join("backups", "new"))
    # results = Results(os.path.join(pwd, "backups", "unfixed_init_2"), "output.csv")
    # results = Results(os.path.join("experiments"))
    results.correlate(correlation_var=["change_rate", "iteration", "duty_cycle","used_teach_count"], grading_var=["AC_fm_integral"],
                      filter_strategy=Strategy(iteration=3))
    results.correlate(correlation_var=["change_rate", "iteration", "duty_cycle","used_teach_count"], grading_var=["AC_fm_integral"])
    results.correlate(correlation_var=["change_rate", "iteration", "duty_cycle","used_teach_count"], grading_var=["AC_fm_integral"],
                      filter_strategy=Strategy(iteration=3,change_rate=0.0))#, exclude_strategy=Strategy(change_rate=1.0))
    # results.plot_scatter(Strategy(place_weights=np.array([1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2])))
    results.plot(filter_strategy=Strategy(), sorting_paramteres="change_rate")
    results.scatter(filter_strategy=Strategy(), sorting_paramteres="change_rate")
    # results.plot_recognition_corelation()
    plt.show()
