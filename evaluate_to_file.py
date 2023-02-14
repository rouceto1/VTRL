#!/usr/bin/env python3
from python.evaluate import *

from python.general_paths import *


def evaluate(_dataset_path, _evaluation_prefix, _evaluation_paths, _weights_file, _GT_file,
             _cache2, use_cache=False, limit=None):
    # make file list against first images (original map)
    images = 143
    if not limit is None:
        images = limit
    file_list = make_file_list(range(7), [0], range(1, images),
                               _dataset_path,
                               _evaluation_prefix, _evaluation_paths)
    gt = read_gt_file(file_list, os.path.join(_evaluation_prefix, _GT_file))
    print("using gt")
    displacements, feature_count, histograms, hist_nn = fm_nn_eval(
        file_list, filetype_NN, filetype_FM, os.path.join(_dataset_path, _weights_file),
        _dataset_path, _cache2, use_cache, gt, limit=limit)
    return file_list, displacements, feature_count, histograms, hist_nn


def evaluate_to_file(_dataset_path, _evaluation_prefix, _evaluation_paths, _weights_file, _GT_file, _eval_out,
                     _cache2, use_cache=False, limit=None):
    out = [evaluate(_dataset_path, _evaluation_prefix, _evaluation_paths, _weights_file, _GT_file,
                    _cache2, use_cache=use_cache, limit=limit)]
    with open(_eval_out, 'wb') as handle:
        pickle.dump(out, handle)
    print("evaluation output at:")
    print(_eval_out)


if __name__ == "__main__":
    evaluate_to_file(dataset_path, evaluation_prefix, evaluation_paths, weights_file, GT_file, eval_out,
                     cache2, use_cache=False, limit=20)
