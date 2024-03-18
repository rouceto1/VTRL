#!/usr/bin/env python3
from python.grading.std_plots import *
from python.grading.results import Results
import matplotlib.pyplot as plt
import matplotlib.cm as cum
import numpy as np
import pandas as pd
import seaborn as sn
import pickle
import os

pwd = os.getcwd()
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
    a = df.groupby(["ee_ratio", "duty_cycle"])["AC_fm_integral"].mean().unstack("duty_cycle")
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
                    m.append(strat.place_weights)
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
    results = Results(os.path.join("backups", "compare"))

    get_preferences(results)

def print_df_to_csv(dfs, path, sorting_parameter, grouping = None):
    #saves infromatio for eahc graphs DF to csv
    #First obtain pairs (touples, things to compare)
    out = []
    separe_datasets = False
    if separe_datasets:
        for d in dfs:
            pass
            #get lines that have same variables
    else:
        pass
    #than seve the csv

    with open():
        pass


def get_graphs_for_paper():
    results = Results(os.path.join("backups", "c_basic"))
    results.add_missions(os.path.join("backups", "c_up_dc_075_025"))
    ## those two: [uptime 025 05 075], [chnge_rate 1 0 -1], [duy_cycle 025 05 075]


    #DC vs changer rate
    #results = Results(os.path.join("backups", "uptime"))
    dfs = scatter_violin(results, filter_strategy=Strategy(iteration=6), exclude_strategy=Strategy(duty_cycle=3.0),variable="AC_fm_integral",
                   sorting_paramteres=["duty_cycle", "change_rate"],
                    plot_params = ["Relative improvement for each strategy over different duty cycles", "Duty cycle", "AC Integral",[0.435,0.475],'lower right']
    )


    scatter_violin(results, filter_strategy=Strategy(iteration=6), exclude_strategy=Strategy(duty_cycle=3.0),variable="AC_fm_integral",
                   sorting_paramteres=["change_rate", "duty_cycle","uptime"], grouping="real_uptime",
                    plot_params = ["Relative improvement for each strategy over different duty cycles", "Duty cycle", "AC Integral",[0.435,0.475],'lower right']
    )

    #metrics
    #results = Results(os.path.join("backups", "metrics_2"))
    #results.add_missions(os.path.join("backups", "metrics"))
    #TODO results.add_missions(os.path.join("backups", "metrics_3"))
    results = Results(os.path.join("backups", "c_methods"))
    ## this one [uptime 025 033], [chnge_rate 1], [duy_cycle 025 033], [method_type 0 1 2]
    scatter_violin(results, filter_strategy=Strategy(iteration=6,roll_data=False, change_rate=1), variable="AC_fm_integral",\
                   ## TODO OLD sorting_paramteres=["metrics_type"],
                   sorting_paramteres=["method_type"],
                   plot_params=["Metrics comparison", "Metrics",
                                "AC Integral",[0.435,0.475]]
                   )

    #eeEEEEEEEEEEEE
    # do NOT use ee only ee2 folders
    #EE vs duty cycle and EE itself
    #results = Results(os.path.join("backups", "ee4"))
    #results.add_missions(os.path.join("backups", "ee5"))
    #TODO results.add_missions(os.path.join("backups", "ee6"))
    #results.add_missions(os.path.join("backups", "ee7"))
    #results.add_missions(os.path.join("backups", "ee8"))
    #results.add_missions(os.path.join("backups", "ee9"))
    #results.add_missions(os.path.join("backups", "ee10"))
    scatter_violin(results, filter_strategy=Strategy(iteration=6, change_rate=1,
                                                     #uptime = 0.25,
                                                     roll_data=False), exclude_strategy=Strategy(),
                  variable="AC_fm_integral", sorting_paramteres=["ee_ratio"],
                   plot_params = ["Exploration/exploitation ratio comparison", "Exploration/exploitation ratio", "AC Integral",[]]
    )
    #scatter_violin(results, filter_strategy=Strategy(iteration=6, change_rate=1, uptime = 0.25,roll_data=False), exclude_strategy=Strategy(),
    #              variable="AC_fm_integral", sorting_paramteres=["duty_cycle", "ee_ratio"])

    scatter_violin(results, filter_strategy=Strategy( change_rate=1,
                                                      #uptime = 0.25,
                                                      roll_data=False, iteration=6), exclude_strategy=Strategy(),
                   variable="AC_fm_integral", sorting_paramteres=[ "duty_cycle","ee_ratio"],
    plot_params = ["Exploration/exploitation ratio progression over different duty cycles", "Duty cycle", "AC Integral",[],'lower right']
    )
    # COMAPRE TO vtrl
    #results = Results(os.path.join("backups", "compare"))

    # results.add_missions(os.path.join("backups", "ee"))
    # compares original vtrl (change_rate = 0, roll_data=True, preteach=true
    # scatter_violin(results, filter_strategy=Strategy(iteration=6, roll_data=True), variable="AC_fm_integral",
    #               sorting_paramteres=["preteach", "change_rate"],
    #               plot_params=["Comparison to VTRL", "Preteach", "AC Integral", [0.435, 0.475], 'lower left']
    #               )
    #results = Results(os.path.join("backups", "c_basic"))
    scatter_violin(results, filter_strategy=Strategy(iteration=6), variable="AC_fm_integral",
                   sorting_paramteres=["change_rate", "preteach", "roll_data"], grouping="roll_pretech",
                   plot_params=["Comparison to VTRL", "Roll data /\nPreteach", "AC Integral", [],
                                'lower left'])


if __name__ == "__main__":
    #get_timigs()

    get_graphs_for_paper()
    #get_progress()

    plt.show()
