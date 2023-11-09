#!/usr/bin/env python3
from python.teach.planner import Mission, Strategy
from python.grading.std_plots import *
import matplotlib.pyplot as plt
import matplotlib.cm as cum
import numpy as np
import pandas as pd
import seaborn as sn
import pickle
import os

pwd = os.getcwd()
import colorsys

strategy_keys = ["uptime", "block_size", "dataset_weights", "used_teach_count", "preferences", "time_limit",
                 "time_advance", "change_rate", "iteration", "duty_cycle", "preteach", "roll_data", "ee_ratio"]
df_keys = ["uptime", "new_param", "block_size", "dataset_weights", "used_teach_count", "preferences", "time_limit",
           "time_advance", "change_rate", "iteration", "duty_cycle", "preteach", "roll_data", "train_time",
           "metrics_type", "train_time", "ee_ratio"]
df_grading_keys = ["AC_fm_integral", "AC_nn_integral", "streak_integral", "AC_fm", "AC_nn"]


def get_only_keys(keys, dictionary):
    out = {}
    for key in dictionary:
        if key in keys:
            out[key] = dictionary[key]
    return out


class Results:
    def __init__(self, path):
        print("-------------------------------------------------")
        self.path = path
        self.missions, self.generator = self.load_missions(path)
        self.output_graph_path = os.path.join(path, "compare.png")
        print(path)
        if self.generator is not None:
            for g in self.generator:
                print(g.replace("\n", ""))
        # for m in self.missions:
        # m.c_strategy.print_parameters()
        # self.t = self.get_N_HexCol(len(self.data))

    def add_missions(self, path):
        print("-------------------------------------------------")
        print(path)
        new, gen2 = self.load_missions(path)
        self.missions.extend(new)
        if gen2 is not None:
            for g in gen2:
                print(g.replace("\n", ""))
        # for m in new:
        # m.c_strategy.print_parameters()

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
        return missions, gen

    def is_strategy_same_as_params(self, strategy, params, exclude):

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
                elif key in ["preferences"]:
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
                elif key in ["preferences"]:
                    if np.array_equal(p[key], m[key]):
                        return False
                elif e[key] == m[key]:
                    return False

        return True

    def get_name_from_strategy_params(self, strategy):
        # return names and values of stategy variables taht are not none

        p = get_only_keys(strategy_keys, vars(strategy))
        name = ""
        for key in p:
            if p[key] is not None:
                if key in ["used_teach_count"]:
                    if p[key] == 0:
                        continue
                name += key + "=" + str(p[key])
        return name

    def concatenate_params(self, strategy, params):
        out = []
        for p in params:
            v = getattr(strategy, p)
            out.append(v)
        strategy.new_param = out








    def filter_strategies(self, mission_params=None, stategy_params=None, exclude_strategy=None, sorting_params=None):

        # dam inclusion param pro mise // napr mise se specifickym polem na startu
        # dam inclusion param pro strategie //napr pouze 3. iterace
        # dam sorting param
        # ono to vyradi a sortne v poradi strategie
        values = []
        strategies = []
        names = []
        override = False
        idx = 0
        for mission in self.missions:
            if len(mission.old_strategies) == 0:
                continue
            if mission.old_strategies[0] == mission_params:
                override = True
            for strategy in mission.old_strategies:
                if not self.is_strategy_same_as_params(strategy, stategy_params, exclude_strategy) or override:
                    continue

                strategy.name = idx
                idx += 1
                if "ee_ratio" in sorting_params:
                    if not hasattr(strategy, "ee_ratio"):
                        if strategy.change_rate == -1:
                            strategy.ee_ratio = 0.0
                        elif strategy.change_rate == 1:
                            strategy.ee_ratio = 1.0
                        else:
                            continue
                    else:
                        if strategy.change_rate == -1:
                            strategy.ee_ratio = 0.0
                if sorting_params == "preferences":
                    if len(values) == 0:
                        values.append(strategy.preferences)
                    is_in_list = np.any(np.all(strategy.preferences == values, axis=1))
                    if not is_in_list:
                        values.append(strategy.preferences)
                elif len(sorting_params) > 1:
                    self.concatenate_params(strategy, sorting_params)
                    if getattr(strategy, "new_param") not in values:
                        values.append(getattr(strategy, "new_param"))
                else:
                    if getattr(strategy, sorting_params[0]) not in values:
                        values.append(getattr(strategy, sorting_params[0]))

                strategies.append(strategy)

        # TODO make this by sortin_params, prolly use getattribute....
        if sorting_params != "preferences" and len(sorting_params) == 1:
            strategies.sort(key=lambda x: getattr(x, sorting_params[0]), reverse=False)
        colors = get_N_HexCol(len(strategies))
        return strategies, colors, values, self.make_pandas_df(strategies, sorting_paramteres=sorting_params)

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

    def correlate(self, filter_strategy=Strategy(), sorting_parameters=["change_rate"], correlation_var=[],
                  grading_var=["AC_fm_integral"], exclude_strategy=Strategy()):
        strategies_to_corelate, colors, values, df = self.filter_strategies(stategy_params=filter_strategy,
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

    def make_pandas_df(self, strategies, sorting_paramteres=None):
        out = []

        for s in strategies:
            a = get_only_keys(df_keys, vars(s))
            b = get_only_keys(df_grading_keys, vars(s.grading[0]))
            out.append(a | b)
        df = pd.DataFrame(out)
        df['preteach'] = df.apply(self.agreagate_preteach, axis=1)
        df['roll_data'] = df.apply(self.agreagate_roll_data, axis=1)
        df['change_rate'] = df.apply(self.name_change_rate, axis=1)
        df["metrics_type"] = df.apply(self.name_metrics, axis=1)
        df['roll_pretech'] = df['roll_data'] + " " + df['preteach']
        df['real_uptime'] = df['uptime'] * df['duty_cycle']
        return df

    def agreagate_preteach(self, dataframe):
        if dataframe["preteach"] == True:
            return "Continued weights"
        else:
            return "Initial weights"

    def agreagate_roll_data(self, dataframe):
        if dataframe["roll_data"] == True:
            return "New data /\n"
        else:
            return "All data /\n"

    def name_change_rate(self, dataframe):
        if dataframe["change_rate"] == 1.0:
            return "Adaptive exploration"
        elif dataframe["change_rate"] == 0.0:
            return "Static exploration"
        elif dataframe["change_rate"] == -1.0:
            return "Random exploration"

    def name_metrics(self,dataframe):
        if dataframe["metrics_type"] == 0:
            return "Enthropy threshold"
        elif dataframe["metrics_type"] == 1:
            return "Enthropy"
        elif dataframe["metrics_type"] == 2:
            return "Ratio"


def comparison_to_old_system():
    results = Results(os.path.join("backups", "compare"))
    #results.add_missions(os.path.join("backups", "ee"))
    # compares original vtrl (change_rate = 0, roll_data=True, preteach=true
    scatter_violin(results, filter_strategy=Strategy(iteration=6, roll_data=True), variable="AC_fm_integral",
                   sorting_paramteres=["preteach", "change_rate"],
                   plot_params = ["Comparison to VTRL", "Preteach", "AC Integral",[],'lower left']
 )
    scatter_violin(results, filter_strategy=Strategy(), variable="AC_fm_integral",
                   sorting_paramteres=["change_rate", "preteach", "roll_data"], grouping="roll_pretech")
    scatter_violin(results, filter_strategy=Strategy(iteration=6, roll_data=True), variable="AC_fm_integral",
                   sorting_paramteres=["roll_data", "change_rate"])
    scatter_violin(results, filter_strategy=Strategy(iteration=6), variable="AC_fm_integral",
                   sorting_paramteres=["change_rate", "preteach", "roll_data"], grouping="roll_pretech")
    scatter_violin(results, filter_strategy=Strategy(iteration=6, roll_data=True, preteach=True),
                   variable="AC_fm_integral",
                   sorting_paramteres=["change_rate"])
    plot_std(results, filter_strategy=Strategy(iteration=6),
             sorting_paramteres=["change_rate", "preteach", "roll_data"])
    plot_std(results, filter_strategy=Strategy(iteration=6, roll_data=True, preteach=True),
             sorting_paramteres=["change_rate"])
    plt.show()

def compute_ee_diff(results,filter_strategy=Strategy(), exclude_strategy=Strategy(), sorting_paramteres=["change_rate"]):
    strategies_to_plot, colors, values, df = results.filter_strategies(stategy_params=filter_strategy,
                                                                       sorting_params=sorting_paramteres,
                                                                       exclude_strategy=exclude_strategy)
    #fig = plt.figure()
    #ax = plt.gca()
    #split df by duty_cycle, for each split compute average AC_fm_integral for each ee_ratio
    #plot this as a function of ee_ratio
    a =df.groupby(["ee_ratio", "duty_cycle"])["AC_fm_integral"].mean().unstack("duty_cycle")
    a.sub(a.T[0]).T.plot(kind="bar")
    #(df.groupby(["ee_ratio", "duty_cycle"])["AC_fm_integral"].median().unstack("duty_cycle").sub().T.plot(kind="bar"))
    plt.axhline(y=0.0, color='r', linestyle=':')
def get_preferences(results):
    preferences = []
    for mission in results.missions:
        m = []
        for strat in mission.old_strategies:
            if strat.change_rate == 1:
                if strat.duty_cycle == 5.0:
                    m.append(strat.preferences)
        if strat.change_rate == 1:
            if strat.duty_cycle == 5.0:
                preferences.append(m)
    plot_preferences(preferences)



def get_dc_vs_change_rate():
    #duty cycle vs change rate
    paths = [item for item in os.listdir("backups") if os.path.isdir(os.path.join("backups", item))]
    #for p in paths:
    #    results = Results(os.path.join("backups", p))
    #    scatter_violin(results,t=p, filter_strategy=Strategy(iteration=6), exclude_strategy=Strategy(),
    #               variable="AC_fm_integral",
    #               sorting_paramteres=["duty_cycle", "change_rate"])
    results = Results(os.path.join("backups", "uptime"))
    scatter_violin(results, filter_strategy=Strategy(iteration=6), exclude_strategy=Strategy(duty_cycle=3.0),variable="AC_fm_integral",
                   sorting_paramteres=["duty_cycle", "change_rate"],
                    plot_params = ["Relative improvement for each strategy over different duty cycles", "Duty cycle", "AC Integral",[0.42,0.48],'lower right']
    )


def get_timigs():
    #calcualtion (and estimation) of time for elarning
    results = Results(os.path.join("backups", "ee2"))
    results.add_missions(os.path.join("backups", "ee3"))
    results.add_missions(os.path.join("backups", "ee4"))
    results.add_missions(os.path.join("backups", "ee5"))
    results.add_missions(os.path.join("backups", "ee"))
    results.add_missions(os.path.join("backups", "metrics"))
    results.add_missions(os.path.join("backups", "metrics_2"))
    results.add_missions(os.path.join("backups", "compare"))
    results.add_missions(os.path.join("backups", "uptime"))
    #scatter(results, filter_strategy=Strategy(roll_data=True), sorting_paramteres=["used_teach_count"])
    scatter_violin(results, filter_strategy=Strategy(roll_data=True), variable="train_time",
                   sorting_paramteres=["uptime", "duty_cycle"])
    scatter_violin(results, filter_strategy=Strategy(roll_data=False), variable="train_time",
                   sorting_paramteres=["uptime", "duty_cycle"])
    scatter(results, filter_strategy=Strategy(roll_data=False, uptime=0.5), sorting_paramteres=["name"])
    stack_violin_iterations(results, filter_strategy=Strategy(iteration=6, roll_data=False), variable="train_time",
                   sorting_paramteres=["uptime", "duty_cycle"])
    stack_violin_iterations(results, filter_strategy=Strategy(iteration=6, roll_data=True), variable="train_time",
                   sorting_paramteres=["uptime", "duty_cycle"])
    #   plot_std_pandas(results, filter_strategy=Strategy(iteration=6, roll_data=False, uptime=0.5), sorting_paramteres=["iteration","uptime"], variable="train_time")


def get_progress():
    #convergence of preferences graph
    results = results = Results(os.path.join("backups", "compare"))
    get_preferences(results)

def get_graphs_for_paper():
    #DC vs changer rate
    results = Results(os.path.join("backups", "uptime"))
    scatter_violin(results, filter_strategy=Strategy(iteration=6), exclude_strategy=Strategy(duty_cycle=3.0),variable="AC_fm_integral",
                   sorting_paramteres=["duty_cycle", "change_rate"],
                    plot_params = ["Relative improvement for each strategy over different duty cycles", "Duty cycle", "AC Integral",[0.435,0.475],'lower right']
    )

    #metrics
    results = Results(os.path.join("backups", "metrics_2"))
    results.add_missions(os.path.join("backups", "metrics"))
    results.add_missions(os.path.join("backups", "metrics_3"))
    scatter_violin(results, filter_strategy=Strategy(iteration=6,roll_data=False, change_rate=1), variable="AC_fm_integral",
                   sorting_paramteres=["metrics_type"],
                   plot_params=["Metrics comparison", "Metrics",
                                "AC Integral",[0.435,0.475]]
                   )

    #eeEEEEEEEEEEEE
    # do NOT use ee only ee2 folders
    #EE vs duty cycle and EE itself
    results = Results(os.path.join("backups", "ee4"))
    results.add_missions(os.path.join("backups", "ee5"))
    results.add_missions(os.path.join("backups", "ee6"))
    results.add_missions(os.path.join("backups", "ee7"))
    results.add_missions(os.path.join("backups", "ee8"))
    results.add_missions(os.path.join("backups", "ee9"))
    results.add_missions(os.path.join("backups", "ee10"))
    scatter_violin(results, filter_strategy=Strategy(iteration=6, change_rate=1, uptime = 0.25,roll_data=False), exclude_strategy=Strategy(),
                  variable="AC_fm_integral", sorting_paramteres=["ee_ratio"],
                   plot_params = ["Exploration/exploitation ratio comparison", "Exploration/exploitation ratio", "AC Integral",[0.435,0.475]]
    )
    #scatter_violin(results, filter_strategy=Strategy(iteration=6, change_rate=1, uptime = 0.25,roll_data=False), exclude_strategy=Strategy(),
    #              variable="AC_fm_integral", sorting_paramteres=["duty_cycle", "ee_ratio"])
    scatter_violin(results, filter_strategy=Strategy( change_rate=1, uptime = 0.25, roll_data=False, iteration=6), exclude_strategy=Strategy(),
                   variable="AC_fm_integral", sorting_paramteres=[ "duty_cycle","ee_ratio"],
    plot_params = ["Exploration/exploitation ratio progression over different duty cycles", "Duty cycle", "AC Integral",[0.435,0.475],'lower right']
    )

    # COMAPRE TO vtrl
    results = Results(os.path.join("backups", "compare"))
    # results.add_missions(os.path.join("backups", "ee"))
    # compares original vtrl (change_rate = 0, roll_data=True, preteach=true
    scatter_violin(results, filter_strategy=Strategy(iteration=6, roll_data=True), variable="AC_fm_integral",
                   sorting_paramteres=["preteach", "change_rate"],
                   plot_params=["Comparison to VTRL", "Preteach", "AC Integral", [0.435,0.475], 'lower left']
                   )
    scatter_violin(results, filter_strategy=Strategy(iteration=6), variable="AC_fm_integral",
                   sorting_paramteres=["change_rate", "preteach", "roll_data"], grouping="roll_pretech",
                   plot_params=["Comparison to VTRL", "Roll data /\nPreteach", "AC Integral", [0.435,0.475], 'lower left'])

if __name__ == "__main__":
    #get_timigs()


    get_graphs_for_paper()
    #get_progress()

    plt.show()
