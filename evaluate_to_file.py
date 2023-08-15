#!/usr/bin/env python3
import numpy as np

from python.evaluate import *

from python.general_paths import *
from scipy.stats import entropy
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd


def get_metric_of_well_understood_image_pairs(histograms, file_list):
    # return list of images that are well understood based on histogram and its enthropy
    # histograms is list of histograms for each image
    # file_list is list of file names for each image
    # return list of file names of images that are well understood

    # get entrhopy of each histogram
    entropies = []
    for h in histograms:
        entropies.append(entropy(h))
    # get mean of entropies
    mean = np.mean(entropies)
    # get std of entropies
    std = np.std(entropies)
    # get list of images that are well understood
    well_understood = []
    bad = []
    for i in range(len(entropies)):
        if entropies[i] < mean + std * 0.0:
            well_understood.append(file_list[i])
        if entropies[i] >= mean + std * 0.0:
            bad.append(file_list[i])
    return well_understood, bad


def parse_given_file_list(file_list):
    # parse given file list and extract image places and seasons
    # file_list is list of tuples(path1,path2
    # get array of [tuples(place, season)]
    # path looks like "'/home/rouceto1/VTRL/datasets/teaching/strands/season_(season_no)/(place_no).png'
    pairs = []
    for fil in range(len(file_list)):
        pair = []
        for j in range(2):
            if "strands" in file_list[fil][j]:
                name = "strands"
            elif "cestlice" in file_list[fil][j]:
                name = "cestlice"
            else:
                name = "all"
            coords = [name, int(file_list[fil][j].split("/")[-1].split(".")[0]),
                      int(file_list[fil][j].split("/")[-2].split("_")[-1])]
            pair.append(coords)
        pairs.append(pair)
    return pairs


def make_confusion_matrix_for_bad_files(bad_files, all_positions, file_pair_list, name, place_weights_contents):
    bad_strands = np.zeros_like(all_positions[1], dtype=float)
    used_strands = np.zeros_like(all_positions[1], dtype=float)
    for i in range(len(bad_files)):
        if bad_files[i][0][0] == "strands":
            bad_strands[bad_files[i][0][2]][bad_files[i][0][1]] += 1.0
            bad_strands[bad_files[i][1][2]][bad_files[i][1][1]] += 1.0
    for i in range(len(file_pair_list)):
        if file_pair_list[i][0][0] == "strands":
            used_strands[file_pair_list[i][0][2]][file_pair_list[i][0][1]] += 1.0
            used_strands[file_pair_list[i][1][2]][file_pair_list[i][1][1]] += 1.0

    bedness_per_image = bad_strands.copy()
    bedness_per_image[used_strands != 0] /= used_strands[used_strands != 0]
    given_strands = np.array(all_positions[1])

    #### info per place
    sum_given_place = np.sum(given_strands, axis=0)  # given IMAGE count per place
    sum_used_place = np.sum(used_strands, axis=0)  # actually used image combinations coutn per place
    sum_bad_place = np.sum(bad_strands, axis=0)  # bad data combinations per place
    sum_w_bad_place = np.sum(bedness_per_image, axis=0)
    badness_per_place = np.sum(bad_strands, axis=0)
    badness_per_place[sum_used_place != 0] /= sum_used_place[sum_used_place != 0]  ##actually used data that is bad

    data_frame1 = pd.DataFrame([place_weights_contents[int(name[-1])], sum_given_place / max(sum_given_place),
                                sum_used_place / max(sum_used_place), badness_per_place],
                               index=["P(used position)", "P(given)", "used", "used bad"],
                               columns=["krajnas", "office street", "office stairs", "kitchen",
                                        "office outside entrance", "sofas outside",
                                        "office outside", "office outisde2"])
    plt.figure()
    sns.heatmap(np.transpose(data_frame1), xticklabels=1, yticklabels=1, annot=True)
    plt.tight_layout()
    plt.savefig(os.path.join(name, "plots", "usage.png"), dpi=800)
    #### plot used positions and weighted bad positions, each bad position is sum how many times/ how many times used
    temp = np.array(all_positions[1]) - 1.0
    times_used = temp + used_strands

    fig, axs = plt.subplots(3)
    fig.suptitle(str(place_weights_contents[int(name[-1])]) + " all times actually used")
    sns.heatmap(np.transpose(times_used), vmin=-1, ax=axs[0])
    plt.title(str(place_weights_contents[int(name[-1])]) + " strands")
    sns.heatmap(np.transpose(bedness_per_image), ax=axs[1])
    t = (np.sum(used_strands, axis=1) != 0)
    filtered_w_bad_s = bedness_per_image[t, :]
    used_seasons = [i for i, x in enumerate(t) if x]
    data_frame = pd.DataFrame(filtered_w_bad_s, index=used_seasons,
                              columns=["krajnas", "office street", "office stairs", "kitchen",
                                       "office outside entrance", "sofas outside", "office outside", "office outisde2"])
    sns.heatmap(np.transpose(data_frame), ax=axs[2], xticklabels=1, yticklabels=1)

    fig.savefig(os.path.join(name, "plots", "usage_heatmap.png"), dpi=800)

    # save data to pickle
    with open(os.path.join(name, "usage.pickle"), 'wb') as handle:
        pickle.dump(data_frame1, handle)

    return
    # plt.show()


def stratagy_mage(old_plan, old_plan_strategy, metrics):

    pass


def make_new_strategy(metrics, old_plan, old_used, out_path):
    # append new infromation based on _chosen_position where after last nonzero collumn is new information
    # modify the original information based on the new metrics
    # save the new information
    old_plan_strategy = load_plan_strategy( out_path)
    new_plan = stratagy_mage( old_plan, old_plan_strategy, metrics)

    return new_plan


def load_plan_strategy( _experiment_path):
    #read json at _experiment_path to get basic params for strategy
    import json
    with open(os.path.join(_experiment_path, "input.json"), 'r') as f:
        params = json.load(f)
    return params

def process_ev_for_training(exp_path, _dataset_path, old_plan,
                            _estimates_in=None,
                            hist_nn=None, conf=None, file_list=None):
    if file_list is None:
        file_list = make_combos_for_teaching(_chosen_positions, _dataset_path,
                                             filetype_FM, conf=conf)


    # read hist_nn from fole
    strategies = [np.array([1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2]),  # outside less
                  np.array([0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0]),  # inside less
                  np.array([1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0]),  # kitchen less
                  ]

    strategies = [np.array([1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                              np.array([0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                              np.array([0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1]),
                              np.array([0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1]),
                              np.array([0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1]),
                              np.array([0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1]),
                              np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1]),
                              np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0]),
                              np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                              ]
    if hist_nn is None:
        with open(_estimates_in, 'rb') as handle:
            hist_nn = pickle.load(handle)
        # get list of images that are well understood
    well_understood_nn, bad_nn = get_metric_of_well_understood_image_pairs(hist_nn, file_list)
    good_list = parse_given_file_list(well_understood_nn)
    bad_list = parse_given_file_list(bad_nn)
    file_pair_list = parse_given_file_list(file_list)
    metrics = make_confusion_matrix_for_bad_files(bad_list, _chosen_positions, file_pair_list, exp_path, strategies)
    #new_strategy = make_new_strategy(metrics, _chosen_positions, file_pair_list, out_path)


def evaluate_for_learning(out_path, _dataset_path, _chosen_positions, _weights_file,
                          _estimates_out=None,
                          _cache2=None, conf=None, file_list=None):
    if file_list is None:
        file_list = make_combos_for_teaching(_chosen_positions, _dataset_path,
                                             filetype_FM, conf=conf)
    hist_nn, displacement = NN_eval(file_list,
                                    out_path, _weights_file,
                                    conf)
    with open(_estimates_out, 'wb') as handle:
        pickle.dump(hist_nn, handle)
    return hist_nn, file_list

def evaluate_for_GT(out_path, _evaluation_prefix, _evaluation_paths, _weights_file, _GT=None,
                    _estimates_out=None,
                    _cache2="/tmp/cache.pkl", conf=None):
    file_list = make_file_list_from_gt(_evaluation_prefix, _GT)
    out = [fm_nn_eval(file_list, filetype_NN, filetype_FM, out_path, _weights_file, _cache2, conf=conf)]
    with open(_estimates_out, 'wb') as handle:
        pickle.dump(out, handle)
    print("GT estiamtes output at:" + str(_estimates_out))
    return out


if __name__ == "__main__":
    config = load_config("NN_config_test.yaml", 512)
    pass
