#!/usr/bin/env python3
from evaluate_to_file import evaluate_to_file, evaluate
from teach import teach
from annotate import annotate
from python.grade_results import *
import argparse
from python.helper_functions import *

pwd = os.getcwd()
parser = argparse.ArgumentParser(
    description='example: --dataset_path "full path" --evaluation_prefix "full path" --weights_folder "full path" '
                '--file_out suffix.picke')
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
parser.add_argument('--weights_file2', type=str, help="name of weights.pt",
                    default="model_tiny.pt")
parser.add_argument('--file_out', type=str, help="name of pickle out",
                    default="_GT_.pickle")
args = parser.parse_args()

filetype_FM = ".bmp"
filetype_NN = ".png"
image_file_template = "place_%d/%05d"
chosen_positions_file = "input.txt"
feature_matcher_file = "FM_out.pickle"

dataset_path = os.path.join(pwd, args.dataset_path)
evaluation_prefix = os.path.join(pwd, args.evaluation_prefix)
evaluation_paths = args.evaluation_paths
annotation_file = args.weights_file
GT_file = os.path.join(evaluation_prefix,"GT.pickle")
weights_file = os.path.join(pwd, args.weights_folder) + annotation_file
weights_file2 = os.path.join(pwd, args.weights_folder) + args.weights_file2
neural_network_file = annotation_file + "_NN_cache.pickle"
cache2 = os.path.join(dataset_path, neural_network_file)
eval_out_file = weights_file + "_eval.pickle"

chosen_positions = np.loadtxt(os.path.join(dataset_path, chosen_positions_file), int)
estimates_out = os.path.join(dataset_path, eval_out_file)
cache = os.path.join(dataset_path, feature_matcher_file)

config = load_config("NN_config_test.yaml", 512)

#LIMIT = 5  # LIMIT allows only first 5-1 images to be evaluated from each season of gathereing
# (there are 3 seasons) It is possible that less images are going to be given since not all pair are in the dataset
if __name__ == "__main__":
    #print("Annotation:")
    #annotate(dataset_path, evaluation_prefix, evaluation_paths,
    #weights_file, GT_file + "_test", cache2, conf=config)
    #print("-------")
    print("Teaching:")
    weights_eval = os.path.join(pwd, args.weights_folder) + "siam_test.pt"
    teach(dataset_path, chosen_positions, weights_eval, cache, conf=config)
    print("-------")
    print("Evaluation to file:")

    #evaluate_to_file(dataset_path, evaluation_prefix, evaluation_paths, weights_eval, GT_file,
    #                 estimates_out + "_tests",
    #                 cache2, config)
    print("-------")
    print("Grading from file:")
    #grade_type(evaluation_prefix, "./tests", estimates_file=estimates_out + "_tests", _GT_file=GT_file + "_test")
    print("-------")
    print("Evaluation straight:")
    # evaluate so it has the same results as GT (diff should be 0)
    estimates = evaluate_to_file(dataset_path, evaluation_prefix, evaluation_paths, weights_eval, GT_file,
                                 estimates_out + "_tests_gt",
                                 cache2, config)
    print("-------")
    print("Grading straight:")
    grade_type(evaluation_prefix, "./tests", _GT_file=GT_file, estimates=estimates)