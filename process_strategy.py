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


dataset_path = os.path.join(pwd, args.dataset_path)
evaluation_paths = args.evaluation_paths
evaluation_prefix = os.path.join(pwd, args.evaluation_prefix)

GT_file = os.path.join(evaluation_prefix,"GT.pickle")
experiments_path = os.path.join(pwd, "experiments/test")
chosen_positions = np.loadtxt(os.path.join(experiments_path, "input.txt"), int)
weights_eval = os.path.join(experiments_path, "weights.pt")
estimates_out = os.path.join(experiments_path, "estimates.pickle")
config = load_config( os.path.join(experiments_path,"NN_config.yaml"), 512)

#LIMIT = 5  # LIMIT allows only first 5-1 images to be evaluated from each season of gathereing
# (there are 3 seasons) It is possible that less images are going to be given since not all pair are in the dataset
if __name__ == "__main__":
    print("Teaching:")
    teach(dataset_path, chosen_positions, weights_eval, conf=config)
    print("-------")
    print("Evaluation straight:")
    # evaluate so it has the same results as GT (diff should be 0)
    estimates = evaluate_to_file(dataset_path, evaluation_prefix, evaluation_paths, weights_eval,
                                 _estimates_out=estimates_out,
                                  conf=config)
    print("-------")
    print("Grading straight:")
    grade_type(experiments_path, _GT_file=GT_file, estimates=estimates)