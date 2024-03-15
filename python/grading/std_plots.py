from python.teach.mission import Mission, Strategy
import colorsys
import os
import seaborn as sn
import matplotlib.pyplot as plt


def get_N_HexCol(N=5):
    HSV_tuples = [(x * 1.0 / (N * 1.5), 1, 1) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out

def pure_plot(ax, df, sorting_paramteres, variable, grouping, plot_params):
    if len(sorting_paramteres) == 2:
        boxplot = sn.boxplot(data=df, x=sorting_paramteres[0], y=variable, hue=sorting_paramteres[1], ax=ax)
    elif len(sorting_paramteres) == 1:
        boxplot = sn.boxplot(data=df, x=sorting_paramteres[0], y=variable, ax=ax)
    elif len(sorting_paramteres) == 3:
        boxplot = sn.boxplot(data=df, x=grouping, hue=sorting_paramteres[0], y=variable, ax=ax)

    for index, info in enumerate(plot_params):
        if index == 0:
            boxplot.set(title=info)
        if index == 1:
            boxplot.set(xlabel=info)
        if index == 2:
            boxplot.set(ylabel=info)
        if index == 3:
            if len(info) == 2:
                ax.set(ylim=(info[0], info[1]))
        if index == 4:
            if len(info) >= 2:
                plt.legend(loc=info)

def scatter_violin(results, filter_strategy=Strategy(), variable="AC_fm_integral", exclude_strategy=Strategy(),
                   sorting_paramteres=["change_rate", "preteach", "roll_data"], grouping="roll_pretech",
                   plot_params=[]):
    stategy_c = filter_strategy
    stategy_s = filter_strategy
    stategy_c.dataset_weights = [1.0, 0.0]
    strategies_to_plot, colors, values, dfc = results.filter_strategies(stategy_params=stategy_c,
                                                                       sorting_params=sorting_paramteres,
                                                                       exclude_strategy=exclude_strategy)
    stategy_s = filter_strategy
    stategy_s.dataset_weights = [0.0, 1.0]
    strategies_to_plot, colors, values, dfs = results.filter_strategies(stategy_params=stategy_s,
                                                                       sorting_params=sorting_paramteres,
                                                                       exclude_strategy=exclude_strategy)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    pure_plot(ax1, dfc, sorting_paramteres, variable, grouping, [plot_params[0]+ "\nCestlice ", "", plot_params[2], [], "upper left"])
    pure_plot(ax2, dfs, sorting_paramteres, variable, grouping, ["Strands", plot_params[1],plot_params[2], [], ""])



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
