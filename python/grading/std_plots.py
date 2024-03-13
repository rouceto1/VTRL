from python.teach.mission import Mission, Strategy
import colorsys
import os
import matplotlib.pyplot as plt
def get_N_HexCol(length):
    HSV_tuples = [(x * 1.0 / (N * 1.5), 1, 1) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out

def scatter_violin(results, filter_strategy=Strategy(), variable="AC_fm_integral", exclude_strategy=Strategy(),
                   sorting_paramteres=["change_rate", "preteach", "roll_data"], grouping="roll_pretech"):
    strategies_to_plot, colors, values, df = results.filter_strategies(stategy_params=filter_strategy,
                                                                       sorting_params=sorting_paramteres,
                                                                       exclude_strategy=exclude_strategy)

def plot_std(results, filter_strategy=Strategy(), exclude_strategy=Strategy(),
             sorting_paramteres=[]):
    fig = plt.figure()
    ax = plt.gca()
    strategies_to_plot, colors, values, df = results.filter_strategies(
                                                                stategy_params=filter_strategy,
                                                                sorting_params=sorting_paramteres,
                                                                exclude_strategy=exclude_strategy)



def plot_preferences(preferences):
    pass

def stack_violin_iterations(results, filter_strategy=Strategy(iteration=6, roll_data=True), exclude_strategy=Strategy(),
                            variable="train_time",
                   sorting_paramteres=["uptime", "duty_cycle"]):
    strategies_to_plot, colors, values, df = results.filter_strategies(stategy_params=filter_strategy,
                                                                       sorting_params=sorting_paramteres,
                                                                       exclude_strategy=exclude_strategy)

def scatter(results, filter_strategy=Strategy(roll_data=False, uptime=0.5), sorting_paramteres=["name"]):
    pass