#!/usr/bin/env python3
from python.evaluate import *

from python.general_paths import *
from scipy.stats import entropy
import seaborn as sns
import matplotlib.pyplot as plt


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
        if entropies[i] < mean + std:
            well_understood.append(file_list[i])
        if entropies[i] >= mean + std:
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
            else:
                name = "cestlice"
            coords = [name, int(file_list[fil][j].split("/")[-1].split(".")[0]),
                      int(file_list[fil][j].split("/")[-2].split("_")[-1])]
            pair.append(coords)
        pairs.append(pair)
    return pairs


def make_confusion_matrix_for_bad_files(bad_files, all_positions):
    cs = all_positions[0] - 1
    st = all_positions[1] - 1
    for i in range(len(bad_files)):
        if bad_files[i][0][0] == "strands":
            st[bad_files[i][0][2]][bad_files[i][1][2]] += 1
        else:
            cs[bad_files[i][0][2]][bad_files[i][1][2]] += 1

    st_1 = st[:, (st != -1).any(axis=0)]
    cs_1 = cs[:, (cs != -1).any(axis=0)]
    sns.heatmap(st_1, vmin=0)
    sns.heatmap(st, vmin=0)

    plt.imshow(st)
    plt.show()
    return [cs, st]


def evaluate_for_learning(out_path, _dataset_path, _chosen_positions, _weights_file,
                          _estimates_out=None,
                          _cache2=None, conf=None, file_list=None):
    if file_list is None:
        file_list = make_combos_for_teaching(_chosen_positions, _dataset_path,
                                             filetype_FM, conf=conf)
    hist_nn, displacement = NN_eval(choose_proper_filetype(filetype_NN, file_list),
                                    out_path, _weights_file,
                                    conf)
    with open(_estimates_out, 'wb') as handle:
        pickle.dump(hist_nn, handle)
    return hist_nn, file_list


def process_ev_for_training(out_path, _dataset_path, _chosen_positions,
                          _estimates_in=None,
                          hist_nn=None, conf=None, file_list=None):
    if file_list is None:
        file_list = make_combos_for_teaching(_chosen_positions, _dataset_path,
                                             filetype_FM, conf=conf)
    #read hist_nn from fole
    if hist_nn is None:
        with open(_estimates_in, 'rb') as handle:
            hist_nn = pickle.load(handle)
        # get list of images that are well understood
    well_understood_nn, bad_nn = get_metric_of_well_understood_image_pairs(hist_nn, file_list)
    good_list = parse_given_file_list(well_understood_nn)
    bad_list = parse_given_file_list(bad_nn)
    make_confusion_matrix_for_bad_files(bad_list, _chosen_positions)


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
