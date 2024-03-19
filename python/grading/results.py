import numpy as np
from python.teach.mission import Mission, Strategy

import pandas as pd
import seaborn as sn
import pickle
import os
from python.grading.std_plots import *

pwd = os.getcwd()

strategy_keys = ["uptime", "block_size", "dataset_weights", "used_teach_count", "preferences", "time_limit",
                 "time_advance", "change_rate", "iteration", "duty_cycle", "preteach", "roll_data", "ee_ratio"]
df_keys_old = ["uptime", "new_param", "block_size", "dataset_weights", "used_teach_count", "preferences", "time_limit",
           "time_advance", "change_rate", "iteration", "duty_cycle", "preteach", "roll_data", "train_time",
           "metrics_type", "train_time", "ee_ratio"]
df_keys = ["uptime", "new_param", "block_size", "dataset_weights", "used_teach_count", "preferences", "time_limit",
           "time_advance", "change_rate", "iteration", "duty_cycle", "preteach", "roll_data", "train_time"
    , "method_type", "train_time", "ee_ratio", "sigma"]
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
                elif key in ["preferences","dataset_weights"]:
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

    def filter_strategies(self, mission_params=None, stategy_params=None, exclude_strategy=None, sorting_params=None, ground_truth_index=0):

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
        return strategies, colors, values, self.make_pandas_df(strategies, ground_truth_index=ground_truth_index)

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

    def make_pandas_df(self, strategies, ground_truth_index=0):
        out = []

        for s in strategies:
            a = get_only_keys(df_keys, vars(s))
            b = get_only_keys(df_grading_keys, vars(s.grading[ground_truth_index])) ## TODO Change this based on cestlice or other, This now only loads one gt for set
            #Probably make it so it has "AC_fm_integral, AC_fm_integral_cestlice, AC_fm_integral_strands" and so on
            out.append(a | b)
        df = pd.DataFrame(out)
        df['preteach'] = df.apply(self.agreagate_preteach, axis=1)
        df['roll_data'] = df.apply(self.agreagate_roll_data, axis=1)
        df['change_rate'] = df.apply(self.name_change_rate, axis=1)
        df["metrics_type"] = df.apply(self.name_metrics, axis=1)
        df["dataset"] = df.apply(self.name_dataset,axis=1)
        df["method_type"] = df["metrics_type"]
        df['roll_pretech'] = df['roll_data'] + " " + df['preteach']
        df['real_uptime'] = df['uptime'] + df['duty_cycle']
        return df

    def agreagate_preteach(self, dataframe):
        if dataframe["preteach"] == True:
            return "Continued weights"
        else:
            return "Initial weights"

    def name_dataset(self, dataframe):
        if dataframe["dataset_weights"][0] == 1 and dataframe["dataset_weights"][1] == 0:
            return "cestlice"
        elif dataframe["dataset_weights"][0] == 0 and dataframe["dataset_weights"][1] == 1:
            return "strands"

    def agreagate_roll_data(self, dataframe):
        if dataframe["roll_data"] == True:
            return "New data"# /\n"
        else:
            return "All data"# /\n"

    def name_change_rate(self, dataframe):
        if dataframe["change_rate"] == 1.0:
            return "Adaptive exploration"
        elif dataframe["change_rate"] == 0.0:
            return "Static exploration"
        elif dataframe["change_rate"] == -1.0:
            return "Random exploration"

    def name_metrics(self, dataframe):
        try:
            if dataframe["metrics_type"] == 0:
                return "Enthropy threshold"
            elif dataframe["metrics_type"] == 1:
                return "Enthropy"
            elif dataframe["metrics_type"] == 2:
                return "Ratio"
        except:
            if dataframe["method_type"] == 0:
                return "Enthropy threshold"
            elif dataframe["method_type"] == 1:
                return "Enthropy"
            elif dataframe["method_type"] == 2:
                return "Ratio"
