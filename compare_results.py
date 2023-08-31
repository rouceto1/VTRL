#!/usr/bin/env python3
from python.teach.planner import Mission, Strategy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import pickle
import os
pwd = os.getcwd()
import colorsys

def get_only_keys(keys, dictionary):
    out = {}
    for key in dictionary:
        if key in keys:
            out[key] = dictionary[key]
    return out


class Results:
    def __init__(self, path):
        self.path = path
        self.missions, self.generator = self.load_missions(path)
        self.output_graph_path = os.path.join(path, "compare.png")
        if self.generator is not None:
            for g in self.generator:
                print(g.replace("\n", ""))
        else:
            for m in self.missions:
                m.c_strategy.print_parameters()
        # self.t = self.get_N_HexCol(len(self.data))

    def add_missions(self,path):
        new, gen2 = self.load_missions(path)
        self.missions.extend(new)
        if gen2 is not None:
            gen2.print()
        else:
            for m in new:
                m.c_strategy.print_parameters()
    def load_missions(self, path):
        experiments_path = os.path.join(pwd, path)
        gen = None
        if os.path.exists(os.path.join(experiments_path, "generator.txt")):
            with open(os.path.join(experiments_path, "generator.txt")) as f:
                gen = f.readlines()
        paths = [item for item in os.listdir(experiments_path) if os.path.isdir(os.path.join(experiments_path, item))]
        paths.sort()
        missions = []
        for mission_name in paths:
            mission = Mission(int(mission_name))
            mission = mission.load(os.path.join(path, mission_name))
            missions.append(mission)
        return missions , gen

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
        override = False
        for mission in self.missions:
            if len(mission.old_strategies) == 0:
                continue
            if mission.old_strategies[0] == mission_params:
                override = True
            for strategy in mission.old_strategies:
                if not self.is_strategy_same_as_params(strategy, stategy_params, exclude_strategy) or override:
                    continue
                if sorting_params == "place_weights":
                    if len(values) == 0:
                        values.append(strategy.place_weights)
                    is_in_list = np.any(np.all(strategy.place_weights == values, axis=1))
                    if not is_in_list:
                        values.append(strategy.place_weights)
                else:
                    if getattr(strategy, sorting_params) not in values:
                        values.append(getattr(strategy, sorting_params))
                strategies.append(strategy)

        # TODO make this by sortin_params, prolly use getattribute....
        if sorting_params != "place_weights":
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

    def plot_std(self, filter_strategy=Strategy(), sorting_paramteres="change_rate",exclude_strategy=Strategy(), mission_params=[]):
        fig = plt.figure()
        ax = plt.gca()
        strategies_to_plot, colors, values = self.filter_strategies(mission_params = mission_params,stategy_params=filter_strategy,
                                                                    sorting_params=sorting_paramteres,
                                                                    exclude_strategy=exclude_strategy)
        GT_versions = ["strands", "grief"]
        data = []
        for v in values:
            data.append([])

        for s, strategy in enumerate(strategies_to_plot):
            grading = strategy.grading
            for grade in grading:
                if grade is None:
                    continue
                if grade.name == "grief":
                    pass
                    continue
                line_fm = grade.AC_fm
                line_nn = grade.AC_nn
                v = getattr(strategy, sorting_paramteres)
                if sorting_paramteres == "place_weights":
                    i = next((i for i, val in enumerate(values) if np.all(val == v)), -1)
                else:
                    i = values.index(v)
                data[i].append(np.array(line_fm).T)
        colors = self.get_N_HexCol(len(values))
        for idx, d in enumerate(data):
            #get std for each d wich is array of [(x,y),..]
            x, y, s = self.interp(*d)
            ax.plot(x, y, label=values[idx], c=colors[idx])
            ax.fill_between(x, y-s,y+s, alpha=0.2, color=colors[idx])


        plt.legend()
        plt.xlim([0, 0.5])
        plt.ylim([0.8, 1])
        plt.title(sorting_paramteres)
        plt.xlabel("allowed image shift")
        plt.ylabel("probability of correct registration")

    def interp(self, *axis_list):
        min_max_xs = [(min(axis[:, 0]), max(axis[:, 0])) for axis in axis_list]

        new_axis_xs = [np.linspace(min_x, max_x, 100) for min_x, max_x in min_max_xs]
        new_axis_ys = [np.interp(new_x_axis, axis[:, 0], axis[:, 1]) for axis, new_x_axis in
                       zip(axis_list, new_axis_xs)]

        midx = [np.mean([new_axis_xs[axis_idx][i] for axis_idx in range(len(axis_list))]) for i in range(100)]
        midy = [np.mean([new_axis_ys[axis_idx][i] for axis_idx in range(len(axis_list))]) for i in range(100)]
        stdx = [np.std([new_axis_ys[axis_idx][i] for axis_idx in range(len(axis_list))]) for i in range(100)]

        #for axis in axis_list:
        #    plt.plot(axis[:, 0], axis[:, 1], c='black')
        #plt.plot(midx, midy, '--', c='black')
        #plt.show()
        return np.array(midx), np.array(midy), np.array(stdx)

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
                    #print(values.index(getattr(strategy, sorting_paramteres)))
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
    results = Results(os.path.join("backups", "new2"))
    #results.add_missions(os.path.join("backups","new2"))

    results.plot_std(filter_strategy=Strategy(), sorting_paramteres="change_rate")
    results.plot_std(filter_strategy=Strategy(iteration=3), sorting_paramteres="change_rate")
    results.plot_std(filter_strategy=Strategy(iteration=3), sorting_paramteres="place_weights")
    results.correlate(correlation_var=["change_rate", "iteration", "duty_cycle", "used_teach_count"], grading_var=["AC_fm_integral"],
                      filter_strategy=Strategy(iteration=3))
    #results.correlate(correlation_var=["change_rate", "iteration", "duty_cycle","used_teach_count"], grading_var=["AC_fm_integral"])
    #results.correlate(correlation_var=["change_rate", "iteration", "duty_cycle","used_teach_count"], grading_var=["AC_fm_integral"],
    #                  filter_strategy=Strategy(iteration=3,change_rate=0.0))#, exclude_strategy=Strategy(change_rate=1.0))
    # results.plot_scatter(Strategy(place_weights=np.array([1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2])))

    #results.plot_std(filter_strategy=Strategy(iteration=0, duty_cycle=4.0), sorting_paramteres="place_weights")
    #results.scatter(filter_strategy=Strategy(), sorting_paramteres="change_rate")
    # results.plot_recognition_corelation()
    plt.show()
