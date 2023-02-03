#!/usr/bin/env python3
from helper_functions import *
from evaluate import *
import argparse
import os
pwd = os.getcwd()
parser = argparse.ArgumentParser(description='example: --dataset_path "full path" --evaluation_prefix "full path" --weights_folder "fullpaht" --file_out suffix.picke')
parser.add_argument('--dataset_path', type=str,
                    help="full path to dataset to be annotated",
                    default="datasets/strands_crop/training_Nov")
parser.add_argument('--evaluation_prefix', type=str, help="path to folder with evaluation sub-folders",
                    default="datasets/strands_crop")
parser.add_argument('--evaluation_paths', type=str, help="names of folders to eval",
                    default=["testing_Dec", "testing_Feb", "testing_Nov"])
parser.add_argument('--weights_folder', type=str, help="path of weights folder",
                    default="weights/")
parser.add_argument('--weights_file', type=str, help="name of weights.pt",
                    default="model_eunord.pt")
parser.add_argument('--file_out', type=str, help="name of pickle out",
                    default="_GT_.pickle")

print("Import functional")
args = parser.parse_args()
dataset_path = os.path.join(pwd,args.dataset_path)
evaluation_prefix = os.path.join(pwd,args.evaluation_prefix)
evaluation_paths = args.evaluation_paths

annotation_file = args.weights_file
GT_file = annotation_file + args.file_out
annotation_weights = os.path.join(pwd,args.weights_folder) + annotation_file

neural_network_file = annotation_file + "_NN_cache.pickle"

cache2 = os.path.join(dataset_path, neural_network_file)
filetype_FM = ".bmp"
filetype_NN = ".png"
image_file_template = "place_%d/%05d"
use_cache = False


def annotate():
    file_list = make_file_list(range(7), [0], range(1, 143), image_file_template,
                               dataset_path, evaluation_prefix,evaluation_paths)

    gt = np.zeros(len(file_list), dtype=np.float64)
    displacements, feature_count, histograms, hist_nn = fm_nn_eval(file_list, filetype_NN, filetype_FM,
                                                                   annotation_weights, dataset_path,
                                                                   cache2, use_cache, gt, False)

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
