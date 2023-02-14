#!/usr/bin/env python3
from evaluate import *
print("Import functional")

dataset_path = "/home/rouceto1/datasets/strands_crop/training_Nov"
evaluation_prefix = "/home/rouceto1/datasets/strands_crop"
evaluation_paths = ["testing_Dec", "testing_Feb", "testing_Nov"]


chosen_positions_file = "input.txt"
weights_file = "exploration.pt"
feature_matcher_file = "FM_out.pickle"

annotation_file = "model_eunord.pt"
GT_file = annotation_file + "_GT_.pickle"
neural_network_file = weights_file + "_NN_cache.pickle"


eval_out_file = weights_file + "_eval.pickle"
eval_out = os.path.join(dataset_path, eval_out_file)

cache = os.path.join(dataset_path, feature_matcher_file)
cache2 = os.path.join(dataset_path, neural_network_file)
filetype_FM = ".bmp"
filetype_NN = ".png"
image_file_template = "place_%d/%05d"
use_cache = False

chosen_positions = np.loadtxt(
    os.path.join(
        dataset_path,
        chosen_positions_file),
    int)


def teach():
    if not os.path.exists(cache) or not use_cache:
        file_list = make_combos_for_teaching(chosen_positions,
                                             dataset_path,
                                             image_file_template,
                                             filetype_FM)

        print("evaluating FM")
        files_with_displacement = fm_eval(file_list, filetype_FM)
        with open(cache, 'wb') as handle:
            pickle.dump(files_with_displacement,
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("making new chance " + cache)
    else:
        print("reading cache " + cache)
        with open(cache, 'rb') as handle:
            files_with_displacement = pickle.load(handle)
    print("Dispalcaments acquired " + cache)
    # print(files_with_displacement)
    # teach NN on all the combinations
    # print (choose_proper_filetype(filetype_NN, files_with_displacement))
    desired_files = np.array(
        choose_proper_filetype(
            filetype_NN,
            files_with_displacement))
    neuralka.NNteach_from_python(desired_files,
                                 "strands",
                                 os.path.join(dataset_path, weights_file),
                                 3)

# EVAL->
# what files to eval


def evaluate():
    # make file list against first images (original map)
    file_list = make_file_list(
        range(7),
        [0],
        range(
            1,
            143),
        image_file_template,
        dataset_path,
        evaluation_prefix,
        evaluation_paths)
    gt = read_gt_file(file_list, os.path.join(evaluation_prefix, GT_file))
    print("using gt:")
    print(gt)
    displacements, feature_count, histograms, hist_nn = fm_nn_eval(
        file_list, filetype_NN, filetype_FM, os.path.join(
            dataset_path, weights_file), dataset_path, cache2, use_cache, gt, True)
    return file_list, displacements, feature_count, histograms, hist_nn


def evaluate_to_file():
    out = [evaluate()]
    with open(eval_out, 'wb') as handle:
        pickle.dump(out,
                    handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("evaluation output at:")
    print(eval_out)


if __name__ == "__main__":
    # teach()
    evaluate_to_file()
