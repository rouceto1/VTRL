#!/usr/bin/env python3
from python.evaluate import *

from python.general_paths import *
from scipy.stats import entropy

def get_metric_of_well_understood_images(histograms, file_list):
    #return list of images that are well understood based on histogram and its enthropy
    #histograms is list of histograms for each image
    #file_list is list of file names for each image
    #return list of file names of images that are well understood

    #get entrhopy of each histogram
    entropies = []
    for h in histograms:
        entropies.append(entropy(h))
    #get mean of entropies
    mean = np.mean(entropies)
    #get std of entropies
    std = np.std(entropies)
    #get list of images that are well understood
    well_understood = []
    bad = []
    for i in range(len(entropies)):
        if entropies[i] < mean + std:
            well_understood.append(file_list[i])
        if entropies[i] >= mean + std:
            bad.append(file_list[i])
    return well_understood, bad


def evaluate_for_learning(out_path, _dataset_path, _chosen_positions, _weights_file,
                          _estimates_out=None,
                          _cache2=None, conf=None, file_list=None):

    if file_list is None:
        file_list  = make_combos_for_teaching(_chosen_positions, _dataset_path,
                                             filetype_FM, conf=conf)
    #compute only NN evaluation and then detect what image pairs are not well estimated based on the returned histogram
    #the return these pairs and compute FM for them too see what can be trained on
    #then train on them and repeat

    hist_nn, displacement = NN_eval(choose_proper_filetype(filetype_NN, file_list),
                                    out_path, _weights_file,
                                    conf)

    #get list of images that are well understood
    well_understood_nn, bad_nn = get_metric_of_well_understood_images(hist_nn, file_list)



    return out


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
