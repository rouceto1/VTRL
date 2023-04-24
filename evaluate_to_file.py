#!/usr/bin/env python3
from python.evaluate import *

from python.general_paths import *


def evaluate(_dataset_path, _evaluation_prefix, _evaluation_paths, _weights_file, _GT_file=None,
             _cache2=None, conf=None):
    # make file list against first images (original map)
    images = 150
    # TODO: limiting things on the file_tile list level. Code in multiple places could be substituted
    if conf["limit"] is not None:
        images = conf["limit"]
    file_list = make_file_list_annotation(range(8), range(1, images), _evaluation_prefix, _evaluation_paths)

    displacements, feature_count, histograms, hist_nn = fm_nn_eval(
        file_list, filetype_NN, filetype_FM, os.path.join(_dataset_path, _weights_file),
        _cache2, conf=conf)
    return file_list, displacements, feature_count, histograms, hist_nn


def evaluate_to_file(_dataset_path, _evaluation_prefix, _evaluation_paths, _weights_file, _GT_file=None, _estimates_out=None,
                     _cache2=None, conf=None):
    out = [evaluate(_dataset_path, _evaluation_prefix, _evaluation_paths, _weights_file, _GT_file,
                    _cache2, conf)]
    with open(_estimates_out, 'wb') as handle:
        pickle.dump(out, handle)
    print("estiamtes output at:" + str(_estimates_out))
    return out


if __name__ == "__main__":
    config = load_config("NN_config_test.yaml", 512)
    evaluate_to_file(dataset_path, evaluation_prefix, evaluation_paths, weights_file, GT_file, eval_out,
                     cache2, config)
