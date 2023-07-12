#!/usr/bin/env python3
from evaluate_to_file import *
from teach import *
from annotate import *
from python.grade_results import *
import argparse
from python.helper_functions import *
from notify_run import Notify
import time

pwd = os.getcwd()
parser = argparse.ArgumentParser(
    description='example: --dataset_path "full path" --evaluation_prefix "full path" --weights_folder "full path" '
                '--file_out suffix.picke')
parser.add_argument('--dataset_path', type=str,
                    help="full path to dataset to teach on",
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

GT_file = os.path.join(evaluation_prefix, "GT_redone_best.pickle")
notify = Notify(endpoint="https://notify.run/cRRiMSUpAEL2LLH37uWZ")


def process(paths, REDO=[True, True, True, True]):
    estimates_grade = None
    file_list_train = None
    for exp in paths:
        start_time = time.time()
        print(exp)
        experiments_path = os.path.join(pwd, "experiments", exp)
        if os.path.exists(os.path.join(experiments_path, "input.pkl")):
            with open(os.path.join(experiments_path, "input.pkl"), 'rb') as handle:
                chosen_positions = pickle.load(handle)
        else:
            chosen_positions = np.loadtxt(os.path.join(experiments_path, "input.txt"), int)
        weights_eval = os.path.join(experiments_path, "weights.pt")
        estimates_grade_out = os.path.join(experiments_path, "estimates.pickle")
        estimates_train_out = os.path.join(experiments_path, "train.pickle")
        config = load_config(os.path.join(pwd, "experiments", "NN_config.yaml"), 512)

        if not os.path.exists(weights_eval) or REDO[0]:
            file_list_train = teach(dataset_path, chosen_positions, experiments_path, conf=config)
        if not os.path.exists(estimates_train_out) or REDO[1]:
            estimates_train = evaluate_for_learning(experiments_path, dataset_path, chosen_positions, weights_eval,
                                                    _estimates_out=estimates_train_out, conf=config,
                                                    file_list=file_list_train)
        if not os.path.exists(estimates_grade_out) or REDO[2]:
            with open(GT_file, 'rb') as handle:
                gt_in = pickle.load(handle)
            estimates_grade = evaluate_for_GT(experiments_path, evaluation_prefix, evaluation_paths, weights_eval,
                                              _GT=gt_in,
                                              _estimates_out=estimates_grade_out, conf=config)
        if not os.path.exists(os.path.join(experiments_path, "input.png")) or REDO[3]:
            with open(GT_file, 'rb') as handle:
                gt_in = pickle.load(handle)
            grade_type(experiments_path, positions=chosen_positions, _GT=gt_in, estimates_file=estimates_grade_out,
                       estimates=estimates_grade, time_elapsed=start_time)
        notify.send('One finished: ' + exp)
    notify.send('Finished')



if __name__ == "__main__":
    REDO = [False,False,True, True]
    process(["empty"], REDO)
