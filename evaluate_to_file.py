#!/usr/bin/env python3
from python.evaluate import *

from python.general_paths import *


def evaluate(_dataset_path, _evaluation_prefix, _evaluation_paths, _weights_file, _GT_file,
             _cache2, use_cache=False, limit=None):
    # make file list against first images (original map)
    images = 143
    # TODO: limiting things on the file_tile list level. Code in multiple places could be substituted
    if limit is not None:
        images = limit
        file_list = make_file_list([0], [0], range(1, images),
                                   _dataset_path, _evaluation_prefix, _evaluation_paths)
    else:
        file_list = make_file_list(range(7), [0], range(1, images),
                                   _dataset_path, _evaluation_prefix, _evaluation_paths)

    displacements, feature_count, histograms, hist_nn = fm_nn_eval(
        file_list, filetype_NN, filetype_FM, os.path.join(_dataset_path, _weights_file),
        _cache2, use_cache)
    return file_list, displacements, feature_count, histograms, hist_nn


def evaluate_to_file(_dataset_path, _evaluation_prefix, _evaluation_paths, _weights_file, _GT_file, _estimates_out,
                     _cache2, use_cache=False, limit=None):
    out = [evaluate(_dataset_path, _evaluation_prefix, _evaluation_paths, _weights_file, _GT_file,
                    _cache2, use_cache=use_cache, limit=limit)]
    with open(_estimates_out, 'wb') as handle:
        pickle.dump(out, handle)
    print("estiamtes output at:" + str(_estimates_out))
    return out


if __name__ == "__main__":
    evaluate_to_file(dataset_path, evaluation_prefix, evaluation_paths, weights_file, GT_file, eval_out,
                     cache2, use_cache=False, limit=20)
