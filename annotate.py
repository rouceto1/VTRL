#!/usr/bin/env python3
from evaluate import *

print("Import functional")

dataset_path = "/home/rouceto1/datasets/strands_crop/training_Nov"
evaluation_prefix = "/home/rouceto1/datasets/strands_crop"
evaluation_paths = ["testing_Dec", "testing_Feb", "testing_Nov"]

annotation_file = "1-2-fast-grief.pt"
GT_file = annotation_file + "_GT_.pickle"
annotation_weights = "/home/rouceto1/git/VTRL/" + annotation_file

neural_network_file = annotation_file + "_NN_cache.pickle"

cache2 = os.path.join(dataset_path, neural_network_file)
filetype_FM = ".bmp"
filetype_NN = ".png"
image_file_template = "place_%d/%05d"
use_cache = False


def annotate():
    file_list = make_file_list(range(7), [0], range(1, 143), image_file_template, dataset_path, evaluation_prefix,
                               evaluation_paths)

    gt = np.zeros(len(file_list), dtype=np.float64)
    displacements, feature_count, histograms, hist_nn = FM_NN_eval(file_list, filetype_NN, filetype_FM,
                                                                   annotation_weights, dataset_path, cache2, use_cache,
                                                                   gt, True)

    annotations = []
    for i in range(len(displacements)):
        if usefull_annotation(feature_count[i], histograms[i]):
            out = [displacements[i]]
        else:
            out = [float('nan')]
        out.append(feature_count[i])
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

    gt_out = os.path.join(evaluation_prefix, GT_file)
    with open(gt_out, 'wb') as handle:
        pickle.dump(annotations, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("GT written " + str(gt_out))


if __name__ == "__main__":
    annotate()
