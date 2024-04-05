from python.teach.mission import Mission, Strategy
import colorsys
import os
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
            continue
        if index == 1:
            boxplot.set(xlabel=info)
            continue
        if index == 2:
            boxplot.set(ylabel=info)
            continue
        if index == 3:
            if len(info) == 2:
                ax.set(ylim=(info[0], info[1]))
            continue
        if index == 4:
            if info is None:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
                continue
            if len(info) >= 2:
                ax.legend(loc=info)
            continue
        if index == 5:
            ax.tick_params(axis='x', labelrotation=info)
            continue

def scatter_violin(results, filter_strategy=Strategy(), variable="AC_fm_integral", exclude_strategy=Strategy(),
                   sorting_paramteres=["change_rate", "preteach", "roll_data"], grouping="roll_pretech",
                   plot_params=[], versions=[1,0,0,1]):
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
    dfs = [dfc0, dfc1, dfs0, dfs1]
    out = []

    if sum(versions) == 4:
        fig, (ax1, ax2) = plt.subplots(2, 2, sharex=True)
        if dfc0 is not None:
            pure_plot(ax1[0], dfc0, sorting_paramteres, variable, grouping, [plot_params[0]+ "\nCestlice cestlice ", "", plot_params[2], [], "upper left"])
            pure_plot(ax2[0], dfs0, sorting_paramteres, variable, grouping, ["Strands cestlice", plot_params[1],plot_params[2], [], ""])
        if dfc1 is not None:
            pure_plot(ax1[1], dfc1, sorting_paramteres, variable, grouping, ["Cestlice strands", "", plot_params[2], [], ""])
            pure_plot(ax2[1], dfs1, sorting_paramteres, variable, grouping, ["Strands strands", plot_params[1],plot_params[2], [], ""])
        out = [dfc0,dfs1]
    elif sum(versions) == 2:
        par =  ["Čestlice",plot_params[1],plot_params[2],plot_params[3],None ,plot_params[5]]
        fig, ax = plt.subplots(1, 2, sharex=False)
        i=0
        for idx, v in enumerate(versions):
            if v == 1:
                if dfs[idx] is None:
                    continue
                pure_plot(ax[i], dfs[idx], sorting_paramteres, variable, grouping, par)
                i+=1
                par = ["Witham Wharf",plot_params[1],"",plot_params[3],plot_params[4],plot_params[5]]
                out.append(dfs[idx])
        fig.tight_layout()
        if len(plot_params) > 0:
            plt.suptitle(plot_params[0])

        #plt.title(plot_params[0])
    return  out

def plot_std(results, filter_strategy=Strategy(), exclude_strategy=Strategy(),
             sorting_paramteres=[]):
    fig = plt.figure()
    ax = plt.gca()
    strategies_to_plot, colors, values, df = results.filter_strategies(
        stategy_params=filter_strategy,
        sorting_params=sorting_paramteres,
        exclude_strategy=exclude_strategy)


def plot_preferences(pref, missions):
    #plot preferences in time manner
    # preferences = [ line1 [ [val1, val2, val3, ... ], [val1, val2, val3, ... ]], line2 [ [val1, val2, val3, ... ], [val1, val2, val3, ... ]], ...]
    #plot error band computed for each line
    #plot mean of each line

    fig , ax = plt.subplots(2)
    mean = []
    std = []
    #if len(pref) < 2:
    #    pref = [pref]
    for data_index, dataset in enumerate([[1,0] ,[0,1]]):
        for index, preferences in enumerate(pref):
            mission = missions[index]
            if (mission.old_strategies[0].dataset_weights == dataset).all():

                for index in range(len(preferences[1])):
                    m = []
                    for p in preferences:
                        m.append(p[index])
                    mean.append(np.mean(m,axis=0))
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

def contour(results, filter_strategy=Strategy(roll_data=False, uptime=0.5),variables="AC_fm_integra", sorting_paramteres=["name"],exclude_strategy=Strategy(),plot_params=[],extremes=True):
    dfs1_a, dfs1_r, dfs1_s, dfc0_a, dfc0_r,dfc0_s = None, None, None, None, None, None
    min = 1
    max = 0
    COUNT=1
    stategy_c = filter_strategy
    stategy_c.dataset_weights = [1.0, 0.0]
    stategy_c.change_rate = 1.0
    strategies_to_plot, colors, values, dfc0_a = results.filter_strategies(stategy_params=stategy_c,
                                                                         sorting_params=sorting_paramteres,
                                                                         exclude_strategy=exclude_strategy,
                                                                         ground_truth_index=0)
    if COUNT >= 2:
        stategy_c.change_rate = -1.0
        strategies_to_plot, colors, values, dfc0_r = results.filter_strategies(stategy_params=stategy_c,
                                                                         sorting_params=sorting_paramteres,
                                                                         exclude_strategy=exclude_strategy,
                                                                ground_truth_index=0)
    if COUNT >= 3:
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
    if COUNT >= 2:
        stategy_s.change_rate = -1.0
        strategies_to_plot, colors, values, dfs1_r = results.filter_strategies(stategy_params=stategy_s,
                                                                         sorting_params=sorting_paramteres,
                                                                         exclude_strategy=exclude_strategy,
                                                                         ground_truth_index=1)
    if COUNT >= 3:
        stategy_s.change_rate = 0.0
        strategies_to_plot, colors, values, dfs1_s = results.filter_strategies(stategy_params=stategy_s,
                                                                         sorting_params=sorting_paramteres,
                                                                         exclude_strategy=exclude_strategy,
                                                                         ground_truth_index=1)
    strands = [dfs1_a, dfs1_r, dfs1_s]

    fig, ax = plt.subplots(COUNT, 2, sharex=False)
    names = ["Čestlice", "Cestlice random","Cestlice static","Witham Wharf", "Strands random", "Strands Static"]
    limits = [[271, 30], [8, 1007]]
    if extremes:
        for index1, dfs in enumerate([cestlice, strands]):
            for index2 in range(COUNT):
                dataframe = dfs[index2].groupby(["uptime", "duty_cycle"]).mean(numeric_only=True)
                #get min and max from datagrame
                dmin = dataframe[variables].min()
                dmax = dataframe[variables].max()
                if dmin < min:
                    min = dmin
                if dmax > max:
                    max = dmax

    print("min: " + str(min) + " max: " + str(max))

    for index1, dfs in enumerate([cestlice, strands]):
        for index2 in range(COUNT):
            dataframe = dfs[index2].groupby(["uptime", "duty_cycle"]).mean(numeric_only=True)
            if COUNT == 1:
                im = plot_contour(dataframe, ax[index1], variables, names[index1 * 3 + index2],
                             extremes=[extremes, min, max], params=plot_params, limits=limits[index1],fig=fig)
                if index1 < 1:
                    ax[index1].set(xlabel="Uptime", ylabel="Duty cycle")
                else:
                    ax[index1].set(xlabel="Uptime")
                ax[index1].set(title=names[index1 * 3 + index2])
                #fig.colorbar(im ,ax = ax[index1])

            else:
                im = plot_contour(dataframe, ax[index2][index1], variables, names[index1*3+index2],
                                  extremes = [extremes, min, max], params = plot_params,limits=limits[index1],fig=fig)
                #fig.colorbar(im, ax = ax[index2][index1])

    fig.tight_layout()
def plot_contour(df, ax, variables,plot_name,extremes,params, limits,fig=None):
    if df is None:
        return
    #get uptime and duty cycle from dfc0 and plot them in a contour plo
    grid_x, grid_y = np.mgrid[0:1:200j, 0:1:200j]
    method = 'cubic'
    #method = 'nearest'
    method = 'linear'

    points = df.index.tolist()
    x = [i[0] for i in points]
    y = [i[1] for i in points]
    print("contour for " + str(len(points)) + " points for " + plot_name)
    print(points)
    grid_data = griddata(list(zip(x, y)), df[variables], (grid_x, grid_y), method=method)

    if extremes[0]:
        im = ax.imshow(grid_data.T, extent=(0, 1, 0, 1), origin='lower', vmin=extremes[1], vmax=extremes[2], label=plot_name )
    else:
        im = ax.imshow(grid_data.T, extent=(0, 1, 0, 1), origin='lower', label=plot_name)
    ax.plot(x,y, 'k.', ms=2)
    labels1 = [item.get_text() for item in ax.get_xticklabels()]
    if "\n" not in labels1[0]:
        texts = []
        for label in labels1:
            text = "{:.0f}\n{:.0f}".format(float(label)*100.0, limits[1]*float(label))
            texts.append(text)
        texts[0] = "0\n0"
        ax.set_xticklabels(texts)
    labels2 = [item.get_text() for item in ax.get_yticklabels()]
    if "\n" in labels2[0]:
        return
    texts = []
    for label in labels2:
        text = "{:.0f}\n{:.0f}".format(float(label)*100.0, limits[0]*float(label))
        texts.append(text)
    texts[0] = "\n\n\n%\n Total"
    ax.set_yticklabels(texts)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    return im
    #abs.set_ylim(0.0, 1.0 * limits[1])

    #abs.set_ylabel("abs")
    #absx.spines["right"].set_position(("axes", 0))
    #abs.set_ylim(ax.get_ylim()[0] * limits[1], ax.get_ylim()[1] * limits[1])
    #pery = ax.twiny()
    #absy = ax.twiny()
    #pery.set_xlabel("%")
    #absy.set_xlabel("abs")
    #absy.spines["right"].set_position(("axes", 1.2))
    #abs.set_xlim(ax.get_xlim()[0] * limits[0], ax.get_xlim()[1] * limits[0])
    #ax.set(xlabel="Uptime", ylabel="duty_cycle")

