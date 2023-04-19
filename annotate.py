#!/usr/bin/env python3
from python.evaluate import *
from python.general_paths import *


def annotate(_dataset_path, _evaluation_prefix, _evaluation_paths,
             _weights_file, _GT_file, _cache2=None, conf=None):
    images = 143
    if conf["limit"] is not None:
        images = conf["limit"]
        file_list = make_file_list([0], [0], range(1, images),
                                   _dataset_path, _evaluation_prefix, _evaluation_paths)
    else:
        file_list = make_file_list_annotation(range(8), range(1, images), _evaluation_prefix, _evaluation_paths)

    displacements, feature_count, histograms, hist_nn = fm_nn_eval(file_list, filetype_NN, filetype_FM,
                                                                   _weights_file, _cache2, conf)
    annotations = []
    for i in range(len(displacements)):
        # if usefull_annotation(feature_count[i], histograms[i]):
        #    out = [displacements[i]]
        # else:
        #    out = [float('nan')]
        out = [displacements[i], feature_count[i]]
        path = os.path.normpath(file_list[i][0])
        split_path = path.split(os.sep)
        out.append(split_path[-3] + "/" + split_path[-2])
        out.append(split_path[-1])
        path2 = os.path.normpath(file_list[i][1])
        split_path2 = path2.split(os.sep)
        out.append(split_path2[-3] + "/" + split_path2[-2])
        out.append(split_path2[-1])
        out.append(histograms[i])
        out.append(hist_nn[i])
        annotations.append(out)
        # if not limit is None:
        #    if limit == i:
        #        print("stopping after " + str(i))
        #        break
    gt_out = os.path.join(_evaluation_prefix, _GT_file)
    with open(gt_out, 'wb') as handle:
        pickle.dump(annotations, handle)
        print("GT written " + str(gt_out))


if __name__ == "__main__":
    weights_file = '/home/rouceto1/git/VTRL/weights/model_eunord.pt'
    GT_file = 'model_eunord.pt_GT_.pickle'

    evaluation_prefix = '/home/rouceto1/git/VTRL/datasets/strands_crop'
    evaluation_paths = ['testing_Dec', 'testing_Feb', 'testing_Nov']
    config = load_config("NN_config_anno.yaml", 512)
    annotate(dataset_path, evaluation_prefix, evaluation_paths,
             weights_file, GT_file, cache2, conf=config)
