#!/usr/bin/env python3

import seaborn as sns
from scipy.stats import entropy

from python.teach.evaluate import *
from python.general_paths import *


def enthropy_std(histograms, file_list):
    entropies = []
    for h in histograms:
        entropies.append(entropy(h),base=len(h))
    mean = np.mean(entropies)
    std = np.std(entropies)
    well_understood = []
    bad = []
    for i in range(len(entropies)):
        if entropies[i] < mean + std * 0.1:
            well_understood.append(file_list[i])
        if entropies[i] >= mean + std * 0.1:
            bad.append(file_list[i])
    return well_understood, bad


def entrhopy_weighted(histograms, file_list):
    files = []
    weights = []
    for i, h in enumerate(histograms):
        files.append(file_list[i])
        weights.append(entropy(h), base=len(h))
    return files, weights


def two_best_ratio(histograms, file_list):
    files = []
    weights = []
    for i, h in enumerate(histograms):
        files.append(file_list[i])
        weights.append(h[1] / h[0])
    return files, weights


def parse_given_file_list(file_list, weights=None):
    sets = []
    for fil in range(len(file_list)):
        set = []
        for j in range(2):
            if "strands" in file_list[fil][j]:
                name = "strands"
            elif "cestlice" in file_list[fil][j]:
                name = "cestlice"
            else:
                name = "all"
            if weights is None:
                coords = [name, int(file_list[fil][j].split("/")[-1].split(".")[0]),
                          int(file_list[fil][j].split("/")[-2].split("_")[-1])]
            else:
                coords = [name,
                          int(file_list[fil][j].split("/")[-1].split(".")[0]),
                          int(file_list[fil][j].split("/")[-2].split("_")[-1]),
                          weights[fil]]

            set.append(coords)
        sets.append(set)
    return sets


def get_listed_files(mission, file_pair_list):
    used_strands = np.zeros_like(mission.c_strategy.plan[1], dtype=float)
    for i in range(len(file_pair_list)):
        if file_pair_list[i][0][0] == "strands":
            used_strands[file_pair_list[i][0][2]][file_pair_list[i][0][1]] += 1.0
            used_strands[file_pair_list[i][1][2]][file_pair_list[i][1][1]] += 1.0
    return used_strands


def get_listed_files_weighted(mission, file_pair_list):
    used_strands = np.zeros_like(mission.c_strategy.plan[1], dtype=float)
    for i in range(len(file_pair_list)):
        if file_pair_list[i][0][0] == "strands":
            used_strands[file_pair_list[i][0][2]][file_pair_list[i][0][1]] += file_pair_list[i][0][3]
            used_strands[file_pair_list[i][1][2]][file_pair_list[i][1][1]] += file_pair_list[i][0][3]
    return used_strands


def calculate_metrics(mission, bad_strands, used_strands):
    sum_used_place = np.sum(used_strands, axis=0)  # actually used image combinations coutn per place

    bedness_per_image = bad_strands.copy()
    bedness_per_image[used_strands != 0] /= used_strands[used_strands != 0]

    given_strands = np.array(mission.c_strategy.plan[1])
    sum_given_place = np.sum(given_strands, axis=0)  # given IMAGE count per place

    badness_per_place = np.sum(bad_strands, axis=0)
    badness_per_place[sum_used_place != 0] /= sum_used_place[sum_used_place != 0]  ##actually used data that is bad

    plot_and_save(mission, sum_given_place, sum_used_place, badness_per_place, used_strands, bedness_per_image)

    return badness_per_place
    # plt.show()


def plot_and_save(mission, sum_given_place, sum_used_place, badness_per_place, used_strands, bedness_per_image):
    data_frame1 = pd.DataFrame([mission.c_strategy.place_weights, sum_given_place / max(sum_given_place),
                                sum_used_place / max(sum_used_place), badness_per_place],
                               index=["weights", "P(given)", "used", "metrics"],
                               columns=["krajnas", "office street", "office stairs", "kitchen",
                                        "office outside entrance", "sofas outside",
                                        "office outside", "office outisde2"])
    plt.figure()
    sns.heatmap(np.transpose(data_frame1), xticklabels=1, yticklabels=1, annot=True)
    plt.tight_layout()
    plt.savefig(os.path.join(mission.c_strategy.metrics_path), dpi=800)
    #### plot used positions and weighted bad positions, each bad position is sum how many times/ how many times used
    temp = np.array(mission.c_strategy.plan[1]) - 1.0
    times_used = temp + used_strands

    fig, axs = plt.subplots(3)
    fig.suptitle("1: times_used, 2: bedness_per_image, 3: times_used/bedness_per_image")
    sns.heatmap(np.transpose(times_used), vmin=-1, ax=axs[0])
    sns.heatmap(np.transpose(bedness_per_image), ax=axs[1])
    t = (np.sum(used_strands, axis=1) != 0)
    filtered_w_bad_s = bedness_per_image[t, :]
    used_seasons = [i for i, x in enumerate(t) if x]
    data_frame = pd.DataFrame(filtered_w_bad_s, index=used_seasons,
                              columns=["krajnas", "office street", "office stairs", "kitchen",
                                       "office outside entrance", "sofas outside", "office outside", "office outisde2"])
    sns.heatmap(np.transpose(data_frame), ax=axs[2], xticklabels=1, yticklabels=1,annot=True, annot_kws={"fontsize":6})

    fig.savefig(os.path.join(mission.c_strategy.usage_path), dpi=800)

    # save data to pickle
    with open(os.path.join(mission.c_strategy.metrics_path) + ".pkl", 'wb') as handle:
        pickle.dump(data_frame1, handle)
    with open(os.path.join(mission.c_strategy.usage_path) + ".pkl", 'wb') as handle:
        pickle.dump(data_frame, handle)


def process_ev_for_training(mission, _dataset_path, old_plan=None,
                            _estimates_in=None,
                            hist_nn=None, conf=None):
    file_list = mission.c_strategy.file_list
    if hist_nn is None:
        with open(mission.c_strategy.estimates_path, 'rb') as handle:
            hist_nn = pickle.load(handle)
        # get list of images that are well understood

    file_pair_list = parse_given_file_list(file_list)  # array of [tuples(place, season)]
    used_strands = get_listed_files(mission, file_pair_list)
    if mission.c_strategy.metrics_type == 0:
        well_understood_nn, bad_nn = enthropy_std(hist_nn, file_list)
        bad_list = parse_given_file_list(bad_nn)  # array of [tuples(place, season)]
        bad_strands = get_listed_files(mission, bad_list)

    elif mission.c_strategy.metrics_type == 1:
        weighted_list, weights = entrhopy_weighted(hist_nn, file_list)
        coordiantes_list = parse_given_file_list(weighted_list, weights)
        bad_strands = get_listed_files_weighted(mission, coordiantes_list)
        print("work1")
    elif mission.c_strategy.metrics_type == 2:
        weighted_list, weights = two_best_ratio(hist_nn, file_list)
        coordiantes_list = parse_given_file_list(weighted_list, weights)
        bad_strands = get_listed_files_weighted(mission, coordiantes_list)
        print("work2")
    metrics = calculate_metrics(mission, bad_strands, used_strands)

    return metrics
