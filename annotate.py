#!/usr/bin/env python3
from python.evaluate import *
from python.general_paths import *


def annotate(_dataset_path, _evaluation_prefix, _evaluation_paths,
             _weights_file, _GT_file, _cache2=None, conf=None):
    images = 143
    if conf["limit"] is not None:
        images = conf["limit"]

    file_list = make_file_list_annotation(range(8), range(1, images), _evaluation_prefix, _evaluation_paths)

    displacements, feature_count_l,feature_count_r, histograms, hist_nn = fm_nn_eval(file_list, filetype_NN, filetype_FM,
                                                                   _weights_file, _cache2, conf)
    annotations = []
    count = 0
    err = 0
    for i in range(len(displacements)):
        if not usefull_annotation(feature_count_l[i],feature_count_r[i], histograms[i]) or abs(displacements[i]) > 2:
            print(feature_count_r[i],feature_count_l[i],displacements[i])
            err+=1
            continue
        out = [displacements[i], feature_count_l[i],feature_count_r[i]]
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
        count +=1
    gt_out = os.path.join(_evaluation_prefix, _GT_file)
    print("GT genreted with entries: " + str(count) + " and failed: " + str(err))
    with open(gt_out, 'wb') as handle:
        pickle.dump(annotations, handle)
        print("GT written " + str(gt_out))


if __name__ == "__main__":
    weights_file = './weights/model_eunord.pt'
    GT_file = 'model_eunord.pt_GT_.pickle'

    evaluation_prefix = './datasets/strands_crop'
    evaluation_paths = ['testing_Dec', 'testing_Feb', 'testing_Nov']
    config = load_config("NN_config_anno.yaml", 512)
    annotate(dataset_path, evaluation_prefix, evaluation_paths,
             weights_file, GT_file, cache2, conf=config)
