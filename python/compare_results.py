import os
import csv
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
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

    def is_strategy_same_as_params(self, strategy, params):

        strategy_keys = ["uptime", "block_size", "dataset_weights", "used_teach_count", "place_weights", "time_limit",
                         "time_advance", "change_rate", "iteration", "duty_cycle"]
        p = get_only_keys(strategy_keys, vars(params))
        m = get_only_keys(strategy_keys, vars(strategy))
        # compare dicts p and m and return true if non None values from p are same in m
        print(p["iteration"], m["iteration"])
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
        return True

    def prepare_plot(self, mission_params=None, stategy_params=None, sorting_params=None):

        # dam inclusion param pro mise // napr mise se specifickym polem na startu
        # dam inclusion param pro strategie //napr pouze 3. iterace
        # dam sorting param
        # ono to vyradi a sortne v poradi strategie
        values = []
        strategies = []
        for mission in self.missions:
            for strategy in mission.old_strategies:
                if not self.is_strategy_same_as_params(strategy, stategy_params):
                    continue
                if getattr(strategy, sorting_params) not in values:
                    values.append(getattr(strategy, sorting_params))
                strategies.append(strategy)

        # TODO make this by sortin_params, prolly use getattribute....
        strategies.sort(key=lambda x: getattr(x, sorting_params), reverse=False)
        colors = self.get_N_HexCol(len(strategies) // 2)
        return strategies, colors, values

    def plot_lines(self, filter_strategy=Strategy()):
        fig = plt.figure()
        ax = plt.gca()
        sort = "change_rate"
        strategies_to_plot, colors, values = self.prepare_plot(stategy_params=filter_strategy, sorting_params=sort)
        styles = ["solid","dashed","dashdot","dotted"]
        GT_versions = ["strands", "grief"]
        for s, strategy in enumerate(strategies_to_plot):
            grading = strategy.grading
            for grade in grading:
                if grade.name == "grief":
                    continue
                style = "solid"
                if len(values) <= len(styles):
                    style = styles[values.index(getattr(strategy, sort))]
                line_fm = grade.AC_fm
                line_nn = grade.AC_nn
                ax.plot(line_fm[0], line_fm[1], label=strategy.title_parameters(), c=colors[s // 2], alpha=0.5,linestyle=style)
                ax.plot(line_nn[0], line_nn[1], label=strategy.title_parameters(), c=colors[s // 2], alpha=0.5,linestyle=style)

        plt.axvline(x=0.035, color='k')
        plt.legend()
        plt.xlim([0, 0.5])
        plt.title(filter_strategy.title_parameters())
        plt.xlabel("allowed image shift")
        plt.ylabel("probability of correct registration")
        plt.show()

    def plot_scatter(self, gt_type="", filter=False, NN=False, both=False):
        fig = plt.figure()
        ax = plt.gca()
        for i, data in enumerate(self.data):
            if not gt_type in self.data[i][self.gt_type]:
                continue
            if float(self.data[i][self.data_counts_filtered]) > 2000:
                pass
                if filter:
                    continue
            if not NN or both:
                ax.scatter(float(self.data[i][self.data_counts_filtered]), float(self.data[i][self.integral_nn]),
                           label=self.data[i][self.names] + self.data[i][self.data_counts_filtered], c=self.t[i],
                           alpha=0.5)
                ax.annotate(self.data[i][self.names] + " NN",
                            (float(self.data[i][self.data_counts_filtered]), float(self.data[i][self.integral_nn])))
            if NN or both:
                ax.scatter(float(self.data[i][self.data_counts_filtered]), float(self.data[i][self.integral]),
                           label=self.data[i][self.names] + self.data[i][self.data_counts_filtered], c=self.t[i],
                           alpha=0.5)
                ax.annotate(self.data[i][self.names],
                            (float(self.data[i][self.data_counts_filtered]), float(self.data[i][self.integral])))
        plt.title(gt_type + str(NN))

        # plt.xlim([-0.1, 20000])
        ax.set_xscale('symlog')
        plt.ylabel("Integral")
        plt.xlabel("Data count")
        # plt.legend()
        # plt.savefig(self.output_graph_path)

    def correlate(self, gt_type=""):
        integral = []
        integral_nn = []
        # for i in ["strands", "grief", "all"]:
        for i in ["strands"]:
            places = np.char.find(self.gt_type_arr, i)
            integral.append(self.integral_arr[places == 0])
            integral_nn.append(self.integral_nn_arr[places == 0])

        corr = [self.data_count_arr, self.data_count_u_arr, self.data_count_u_arr - self.data_count_arr,
                self.percentages]

        # self.percentages * self.strands, self.percentages * self.cestlice, self.strands]
        names = ["data given (DG)", "data used", "data rejected", "DG from whole",
                 # "DG strands", "DG cestlice", "strands/cestlice",
                 "strands",
                 "strands nn"
                 ]
        # "strands", "grief", "ALL",
        # "strands nn", "grief nn", "ALL nn"]
        # get only corr where gt_type is not in gt_type_arr
        corr = np.transpose(corr)
        corr = corr[places == 0]
        corr = np.transpose(corr)
        corr = np.concatenate((corr, integral))
        corr = np.concatenate((corr, integral_nn))
        R2 = np.corrcoef(corr)
        df_cm = pd.DataFrame(R2, index=[i for i in names],
                             columns=[i for i in names])
        plt.figure()
        plt.title(gt_type)
        sn.heatmap(df_cm, annot=True)

    def plot_lines_old(self, gt_type="", NN=False, both=False):
        # loads and plots AC_lines from line.pkl for each folder
        fig = plt.figure()
        ax = plt.gca()
        for i, data in enumerate(self.data):
            if not gt_type in self.data[i][self.gt_type]:
                continue
            if float(self.data[i][self.data_counts_filtered]) > 2000:
                pass
                # continue
            if NN or both:
                type = "line_"
                with open(os.path.join(self.path, self.data[i][self.names], type + gt_type + ".pkl"), 'rb') as f:
                    line = pickle.load(f)
                ax.plot(line[0], line[1], label=self.data[i][self.integral], c=self.t[i], alpha=0.5)
            if not NN or both:
                type = "line_NN_"
                with open(os.path.join(self.path, self.data[i][self.names], type + gt_type + ".pkl"), 'rb') as f:
                    line = pickle.load(f)
                ax.plot(line[0], line[1], label=self.data[i][self.integral_nn] + " NN", c=self.t[i], alpha=0.5)

        plt.axvline(x=0.035, color='k')
        plt.legend()
        # plt.ylim([0.5, 1])
        plt.xlim([0, 0.5])
        plt.title(gt_type + str(NN))
        plt.xlabel("allowed image shift")
        plt.ylabel("probability of correct registration")

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
    results = Results(os.path.join("backups", "full_run"))
    # results = Results(os.path.join(pwd, "backups", "unfixed_init_2"), "output.csv")
    # results = Results(os.path.join("experiments"))
    # results.correlate()
    # results.plot_scatter(Strategy(place_weights=np.array([1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2])))
    results.plot_lines(Strategy())
    # results.plot_recognition_corelation()
    plt.show()
