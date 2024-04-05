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
    missions = []
    for mission in results.missions:
        m = []
        for strat in mission.old_strategies:
            if strat.change_rate == 1:
                if strat.duty_cycle == 0.5:
                    m.append(strat.preferences)
        if len(m) > 0:
            if strat.change_rate == 1:
                if strat.duty_cycle == 0.5:
                    preferences.append(m)
                    missions.append(mission)
    plot_preferences(preferences, missions)

def get_pref_df(results,filter_strategy=Strategy(), exclude_strategy=Strategy(), sorting_paramteres=["iteration"],plot_params = [""]):
    stategy_c = filter_strategy
    stategy_c.dataset_weights = [1.0, 0.0]
    strategies_to_plot, colors, values, dfc = results.filter_strategies(stategy_params=stategy_c,
                                                                       sorting_params=sorting_paramteres,
                                                                       exclude_strategy=exclude_strategy, ground_truth_index=0)
    stategy_s = filter_strategy
    stategy_s.dataset_weights = [0.0, 1.0]
    strategies_to_plot, colors, values, dfs = results.filter_strategies(stategy_params=stategy_s,
                                                                       sorting_params=sorting_paramteres,
                                                                       exclude_strategy=exclude_strategy, ground_truth_index=0)
    fig, ax = plt.subplots(2)
    for idx, df in enumerate([dfc, dfs]):
        if df is None:
            continue
        plot_pref_df(df, ax[idx],idx,plot_params)
    plt.title(plot_params[0])

    plt.show()

    #for each place

def plot_pref_df(df,ax,data,plot_params):
    # for each place and iteration get mean and std of preferences
    # plot this as a function of place and iteration
    d = [r.preferences [data] for i, r in df.iterrows()]
    df2 = pd.DataFrame.from_records(d)
    df2 = df2.assign(iter=df.iteration).groupby("iter")
    mean = df2.mean()
    std = df2.std(ddof=1)

    for i in range(mean.shape[1]):
        iter = range(mean.shape[0])
        ax.plot(iter, mean[i], label=i)
        ax.fill_between(iter, mean[i] - std[i], mean[i] + std[i], alpha=0.2)


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


    #results = Results(os.path.join("backups", "c_basic_sigma"))
    #results.add_missions(os.path.join("backups", "c_methods")).
    #results.add_missions(os.path.join("backups", "c_methods_sigma"))# DONE
    #results.add_missions(os.path.join("backups", "c_metrics_c"))# DONE
    #results.add_missions(os.path.join("backups", "c_compare_sigma"))
    #results.add_missions(os.path.join("backups", "c_compare_c"))
    #results.add_missions(os.path.join("backups", "c_ee_2"))
    #results.add_missions(os.path.join("backups", "c_ee_3"))
    #results.add_missions(os.path.join("backups", "c_ee_c"))
    names = ["c_basic_sigma", "c_methods", "c_methods_sigma", "c_metrics_c", "c_compare_sigma", "c_compare_c", "c_ee_2", "c_ee_3", "c_ee_c"]

    for name in names:
        results = Results(os.path.join("backups", name))
        get_pref_df(results, filter_strategy=Strategy(change_rate=1, roll_data=False, uptime = 0.5, preteach=True, m_type=0, ee_ratio=1.0),plot_params=[name])
    #get_preferences(results)

def print_df_to_csv(dfs, path, sorting_parameter):
    #saves infromatio for eahc graphs DF to csv
    #First obtain pairs (touples, things to compare)
    out = []
    separe_datasets = False
    #makes array of "AC_fm_integral" for combination in dataframes
    possible = ["block_size", "dataset", "iteration", "duty_cycle", "change_rate", "preteach", "roll_data", "method_type", "ee_ratio", "iteration", "real_uptime"]

    if 'sigma' in dfs[0].head():
        possible.append("sigma")
    else:
        print("THIS DOES NOT HAVE SIGMA")

    sort = []
    for p in possible:
        if p not in sorting_parameter:
            sort.append(p)
    for p in possible:
        if p not in sort:
            sort.append(p)
    sort.reverse()
    out = []
    for d in dfs:
        out.append(d.groupby(sort)["AC_fm_integral"].mean())

    for idx, o in enumerate(out):
        o.to_csv(path + str(idx)+'.csv')

def get_basic():
    #results = Results(os.path.join("backups", "c_basic"))
    results = Results(os.path.join("backups", "c_up_dc_075_025"))
    results.add_missions(os.path.join("backups", "c_basic_sigma"))
    results.add_missions(os.path.join("backups", "c_up_dc_sweep"))
    results.add_missions(os.path.join("backups", "c_up_dc_sweep_2"))
    results.add_missions(os.path.join("backups", "c_up_dc_sweep_3"))

    results.add_missions(os.path.join("backups", "c_up_dc_sweep_c2"))
    results.add_missions(os.path.join("backups", "c_up_dc_sweep_c3"))
    results.add_missions(os.path.join("backups", "c_sweep_s"))
    results.add_missions(os.path.join("backups", "c_sweep_c4"))
    results.add_missions(os.path.join("backups", "c_sweeep_5"))

    results.add_missions(os.path.join("backups", "c_sweep_6"))
    results.add_missions(os.path.join("backups", "c_sweep_7"))
    results.add_missions(os.path.join("backups", "c_sweep_8"))
    ## those two: [uptime 025 05 075], [chnge_rate 1 0 -1], [duy_cycle 025 05 075]
    # results.add_missions(os.path.join("backups", "c_up_dc_sweep_c"))
    # DC vs changer rate
    # results = Results(os.path.join("backups", "uptime"))
    #dfs = scatter_violin(results, filter_strategy=Strategy(),
    #                     variable="AC_fm_integral",
    #                     sorting_paramteres=["duty_cycle", "change_rate"],
    #                     plot_params=["Relative improvement for each strategy over different duty cycles", "Duty cycle",
    #                                  "AC Integral", [0.435, 0.475], 'lower right']
    #                     )


    #name = "7_"
    #print_df_to_csv(dfs, pwd + "/datafast/2024_ral_predictive_roura/" + name, sorting_parameter=["change_rate", "real_uptime"])

    contour(results, filter_strategy=Strategy(roll_data=False, preteach=True),
                         variables="AC_fm_integral",
                         sorting_paramteres=["change_rate", "duty_cycle", "uptime"],
                         plot_params=["Relative improvement for each strategy over different duty cycles", "real_uptime",
                                      "AC Integral", [0.435, 0.475], 'lower right'],extremes=False)
    contour(results, filter_strategy=Strategy(iteration=6, roll_data=False, preteach=True),
                         variables="train_time",
                         sorting_paramteres=["change_rate", "duty_cycle", "uptime"],
                         plot_params=["time for each strategy over different duty cycles", "real_uptime",
                                      "train_time", [0.435, 0.475], 'lower right'],extremes=False)

    contour(results, filter_strategy=Strategy(iteration=6, roll_data=False, preteach=True),
                         variables="used_teach_count",
                         sorting_paramteres=["change_rate", "duty_cycle", "uptime"],
                         plot_params=["Rime for each strategy over different duty cycles", "real_uptime",
                                      "used_teach_count", [0.435, 0.475], 'lower right'])
def get_metrics(name = "6_"): ## THIS IS DONE

    #metrics
    #results = Results(os.path.join("backups", "metrics_2"))
    #results.add_missions(os.path.join("backups", "metrics"))
    #results.add_missions(os.path.join("backups", "metrics_3"))
    results = Results(os.path.join("backups", "c_methods"))
    results.add_missions(os.path.join("backups", "c_methods_sigma"))# DONE
    results.add_missions(os.path.join("backups", "c_metrics_c"))# DONE
    #TODO use iteration 6,
    ## this one [uptime 025 033], [chnge_rate 1], [duy_cycle 025 033], [method_type 0 1 2]
    dfs = scatter_violin(results, filter_strategy=Strategy(roll_data=False, change_rate=1), exclude_strategy=Strategy(),variable="AC_fm_integral",\
                   ## TODO OLD sorting_paramteres=["metrics_type"],
                   sorting_paramteres=["method_type"],
                   plot_params=["", "",
                                "AC Integral", [],'lower left',0]
                   )




    print_df_to_csv(dfs,pwd + "/datafast/2024_ral_predictive_roura/" + name, sorting_parameter=["method_type"])
def get_ee(name = "8_"):
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
    results = Results(os.path.join("backups", "c_ee"))
    results = Results(os.path.join("backups", "c_ee_2"))

    results.add_missions(os.path.join("backups", "c_ee_3"))
    results.add_missions(os.path.join("backups", "c_ee_4"))
    results.add_missions(os.path.join("backups", "c_ee_5"))
    results.add_missions(os.path.join("backups", "c_ee_c"))
    ## TODO NEEDS MORE DATA
    dfs = scatter_violin(results, filter_strategy=Strategy(change_rate=1,
                                                     #uptime = 0.25,
                                                     roll_data=False), exclude_strategy=Strategy(iteration=0),
                  variable="AC_fm_integral", sorting_paramteres=["ee_ratio"],
                   plot_params = ["", "Exploration/exploitation ratio", "AC Integral",[],'lower left',45]
    )

    print_df_to_csv(dfs, pwd + "/datafast/2024_ral_predictive_roura/" + name, sorting_parameter=["ee_ratio"])
    #scatter_violin(results, filter_strategy=Strategy(iteration=6, change_rate=1, uptime = 0.25,roll_data=False), exclude_strategy=Strategy(),
    #              variable="AC_fm_integral", sorting_paramteres=["duty_cycle", "ee_ratio"])

def get_compare():

    # COMAPRE TO vtrl
    #results = Results(os.path.join("backups", "compare"))

    # results.add_missions(os.path.join("backups", "ee"))
    # compares original vtrl (change_rate = 0, roll_data=True, preteach=true
    # scatter_violin(results, filter_strategy=Strategy(iteration=6, roll_data=True), variable="AC_fm_integral",
    #               sorting_paramteres=["preteach", "change_rate"],
    #               plot_params=["Comparison to VTRL", "Preteach", "AC Integral", [0.435, 0.475], 'lower left']
    #               )
    #results = Results(os.path.join("backups", "c_basic"))
    results = Results(os.path.join("backups", "c_compare_sigma"))

    results.add_missions(os.path.join("backups", "c_compare_c"))

    results.add_missions(os.path.join("backups", "c_compare_2"))
    results.add_missions(os.path.join("backups", "c_compare_4"))
    #results.add_missions(os.path.join("backups", "c_compare_5"))
    dfs = scatter_violin(results,exclude_strategy=Strategy(iteration=0),filter_strategy=Strategy(), variable="AC_fm_integral",
                   sorting_paramteres=["change_rate", "preteach", "roll_data"], grouping="roll_pretech",
                   plot_params=["", "", "AC Integral", [],
                                'lower left',45] , versions=[1,0,0,1])

    name = "rp_"
    print_df_to_csv(dfs, pwd + "/datafast/2024_ral_predictive_roura/" + name,
                    sorting_parameter=["change_rate", "preteach", "roll_data"])

    dfs = scatter_violin(results,exclude_strategy=Strategy(iteration=0),filter_strategy=Strategy(roll_data=False), variable="AC_fm_integral",
                   sorting_paramteres=["change_rate", "preteach", "roll_data"], grouping="roll_pretech",
                   plot_params=["", "", "AC Integral", [],
                                'lower left',0] , versions=[1,0,0,1])
    name = "cr_"
    print_df_to_csv(dfs,pwd + "/datafast/2024_ral_predictive_roura/" + name, sorting_parameter=["change_rate", "preteach", "roll_data"])
    #dfs = scatter_violin(results,filter_strategy=Strategy(iteration=6), exclude_strategy=Strategy(iteration=0), variable="AC_fm_integral",
    ##               sorting_paramteres=["change_rate", "preteach", "roll_data"], grouping="roll_pretech",
    #               plot_params=["Comparison to VTRL 6", "Roll data /\nPreteach", "AC Integral", [],
    #                            'lower left'], versions=[1,0,0,1])

def get_graphs_for_paper():
    get_basic()
    #get_metrics()
    #get_ee()
    #get_compare()



if __name__ == "__main__":

    #get_timigs()
    get_graphs_for_paper()
    #get_progress()

    plt.show()
