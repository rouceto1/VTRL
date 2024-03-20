from python.teach.mission import Mission, Strategy
import colorsys
import os
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
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
                   plot_params=[], all_four=True):
    stategy_c = filter_strategy
    stategy_c.dataset_weights = [1.0, 0.0]
    strategies_to_plot, colors, values, dfc0 = results.filter_strategies(stategy_params=stategy_c,
                                                                       sorting_params=sorting_paramteres,
                                                                       exclude_strategy=exclude_strategy, ground_truth_index=0)
    strategies_to_plot, colors, values, dfc1 = results.filter_strategies(stategy_params=stategy_c,
                                                                       sorting_params=sorting_paramteres,
                                                                       exclude_strategy=exclude_strategy, ground_truth_index=1)
    stategy_s = filter_strategy
    stategy_s.dataset_weights = [0.0, 1.0]
    strategies_to_plot, colors, values, dfs0 = results.filter_strategies(stategy_params=stategy_s,
                                                                       sorting_params=sorting_paramteres,
                                                                       exclude_strategy=exclude_strategy, ground_truth_index=0)
    strategies_to_plot, colors, values, dfs1 = results.filter_strategies(stategy_params=stategy_s,
                                                                       sorting_params=sorting_paramteres,
                                                                       exclude_strategy=exclude_strategy, ground_truth_index=1)
    if all_four:
        fig, (ax1, ax2) = plt.subplots(2, 2, sharex=True)
        pure_plot(ax1[0], dfc0, sorting_paramteres, variable, grouping, [plot_params[0]+ "\nCestlice cestlice ", "", plot_params[2], [], "upper left"])
        pure_plot(ax2[0], dfs0, sorting_paramteres, variable, grouping, ["Strands cestlice", plot_params[1],plot_params[2], [], ""])

        pure_plot(ax1[1], dfc1, sorting_paramteres, variable, grouping, ["Cestlice strands", "", plot_params[2], [], ""])
        pure_plot(ax2[1], dfs1, sorting_paramteres, variable, grouping, ["Strands strands", plot_params[1],plot_params[2], [], ""])
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        pure_plot(ax1, dfc0, sorting_paramteres, variable, grouping, [plot_params[0]+ "\nCestlice cestlice ", "", plot_params[2], [], "upper left"])
        pure_plot(ax2, dfs1, sorting_paramteres, variable, grouping, ["Strands strands", plot_params[1],plot_params[2], [], ""])
    return [dfc0,dfs1]

def plot_std(results, filter_strategy=Strategy(), exclude_strategy=Strategy(),
             sorting_paramteres=[]):
    fig = plt.figure()
    ax = plt.gca()
    strategies_to_plot, colors, values, df = results.filter_strategies(
        stategy_params=filter_strategy,
        sorting_params=sorting_paramteres,
        exclude_strategy=exclude_strategy)


def plot_preferences(pref):
    #plot preferences in time manner
    # preferences = [ line1 [ [val1, val2, val3, ... ], [val1, val2, val3, ... ]], line2 [ [val1, val2, val3, ... ], [val1, val2, val3, ... ]], ...]
    #plot error band computed for each line
    #plot mean of each line

    fig , ax = plt.subplots(2)
    mean = []
    std = []
    if len(pref) != 2:
        pref = [pref]
    for preferences in pref:
        for index in range(len(preferences[1])):
            m = []
            for p in preferences:
                m.append(p[index])
            mean.append(np.mean(m ,axis=0))
            std.append(np.std(m,axis=0))
        iter = range(len(mean))
        plt.legend(mean, iter)
        mean = np.array(mean)
        std = np.array(std)
        ax[0].plot(iter, mean,"x",linestyle="-")
        for i in range(len(mean)):
            ax[0].fill_between(iter, mean.T[i] - std.T[i], mean.T[i] + std.T[i], alpha=0.2)
        plt.show()




def stack_violin_iterations(results, filter_strategy=Strategy(iteration=6, roll_data=True), exclude_strategy=Strategy(),
                            variable="train_time",
                            sorting_paramteres=["uptime", "duty_cycle"]):
    strategies_to_plot, colors, values, df = results.filter_strategies(stategy_params=filter_strategy,
                                                                       sorting_params=sorting_paramteres,
                                                                       exclude_strategy=exclude_strategy)


def scatter(results, filter_strategy=Strategy(roll_data=False, uptime=0.5), sorting_paramteres=["name"]):
    pass

def contour(results, filter_strategy=Strategy(roll_data=False, uptime=0.5),variables="AC_fm_integra", sorting_paramteres=["name"],exclude_strategy=Strategy(),plot_params=[]):
    stategy_c = filter_strategy
    stategy_c.dataset_weights = [1.0, 0.0]
    stategy_c.change_rate = 1.0
    strategies_to_plot, colors, values, dfc0_a = results.filter_strategies(stategy_params=stategy_c,
                                                                         sorting_params=sorting_paramteres,
                                                                         exclude_strategy=exclude_strategy,
                                                                         ground_truth_index=0)
    stategy_c.change_rate = -1.0
    strategies_to_plot, colors, values, dfc0_r = results.filter_strategies(stategy_params=stategy_c,
                                                                         sorting_params=sorting_paramteres,
                                                                         exclude_strategy=exclude_strategy,
                                                                          ground_truth_index=0)
    stategy_c.change_rate = 0.0
    strategies_to_plot, colors, values, dfc0_s = results.filter_strategies(stategy_params=stategy_c,
                                                                         sorting_params=sorting_paramteres,
                                                                         exclude_strategy=exclude_strategy,
                                                                         ground_truth_index=0)
    cestlice = [dfc0_a, dfc0_r,dfc0_s]

    stategy_s = filter_strategy
    stategy_s.dataset_weights = [0.0, 1.0]
    stategy_s.change_rate = 1.0
    strategies_to_plot, colors, values, dfs1_a = results.filter_strategies(stategy_params=stategy_s,
                                                                         sorting_params=sorting_paramteres,
                                                                         exclude_strategy=exclude_strategy,
                                                                         ground_truth_index=1)
    stategy_s.change_rate = -1.0
    strategies_to_plot, colors, values, dfs1_r = results.filter_strategies(stategy_params=stategy_s,
                                                                         sorting_params=sorting_paramteres,
                                                                         exclude_strategy=exclude_strategy,
                                                                         ground_truth_index=1)
    stategy_s.change_rate = 0.0
    strategies_to_plot, colors, values, dfs1_s = results.filter_strategies(stategy_params=stategy_s,
                                                                         sorting_params=sorting_paramteres,
                                                                         exclude_strategy=exclude_strategy,
                                                                         ground_truth_index=1)
    strands = [dfs1_a, dfs1_r, dfs1_s]

    fig, ax = plt.subplots(3, 2, sharex=True)
    names = ["Cestlice active", "Cestlice random","Cestlice static","Strands active", "Strands random", "Strands Static"]
    limits = [[30, 271],[1007, 8]]
    for index1, dfs in enumerate([cestlice, strands]):
        for index2, df in enumerate(dfs):
            plot_contour(df, ax[index2][index1], variables, names[index1*3+index2],limits=limits[index1])

def plot_contour(df, ax, variables,plot_name,limits):
    #get uptime and duty cycle from dfc0 and plot them in a contour plo
    grid_x, grid_y = np.mgrid[0:1:200j, 0:1:200j]
    method = 'cubic'
    #method = 'nearest'
    #method = 'linear'
    dataframe = df.groupby(["uptime", "duty_cycle"]).mean()
    points = dataframe.index.tolist()
    x = [i[0] for i in points]
    y = [i[1] for i in points]
    print("contour for " + str(len(points)) + " points for " + plot_name)
    print(points)
    grid_data = griddata(list(zip(x, y)), dataframe[variables], (grid_x, grid_y), method=method)
    ax.imshow(grid_data.T, extent=(0,1, 0,1), origin='lower')
    ax.plot(x,y, 'k.', ms=2)
    ax.set(title=plot_name)