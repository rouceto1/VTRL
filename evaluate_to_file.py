#!/usr/bin/env python3
from python.evaluate import *

from python.general_paths import *

def evaluate_for_learning(out_path, _dataset_path, _chosen_positions, _weights_file,
                          _estimates_out=None,
                          _cache2=None, conf=None, file_list=None):

    if file_list is None:
        file_list = file_list = make_combos_for_teaching(_chosen_positions, _dataset_path,
                                             filetype_FM, conf=conf)
    out = [fm_nn_eval(file_list, filetype_NN, filetype_FM, out_path, _weights_file, _cache2, conf=conf)]
    with open(_estimates_out, 'wb') as handle:
        pickle.dump(out, handle)
    print("GT estiamtes output at:" + str(_estimates_out))
    return out


def evaluate_for_GT(out_path, _evaluation_prefix, _evaluation_paths, _weights_file, _GT=None,
                    _estimates_out=None,
                    _cache2=None, conf=None):
    file_list = make_file_list_from_gt(_evaluation_prefix, _GT)
    out = [fm_nn_eval(file_list, filetype_NN, filetype_FM, out_path, _weights_file, _cache2, conf=conf)]
    with open(_estimates_out, 'wb') as handle:
        pickle.dump(out, handle)
    print("GT estiamtes output at:" + str(_estimates_out))
    return out


if __name__ == "__main__":
    config = load_config("NN_config_test.yaml", 512)
    pass
