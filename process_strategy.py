#!/usr/bin/env python3
from evaluate_to_file import *
from teach import *
from annotate import *
from python.grade_results import *
import argparse
from python.helper_functions import *
import time
from planner import Mission

import logging
import warnings

logger = logging.getLogger()
old_level = logger.level
logger.setLevel(100)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
    description='example: --dataset_path "full path" --evaluation_prefix "full path" --weights_folder "full path" '
                '--file_out suffix.picke')
parser.add_argument('--dataset_path', type=str, help="full path to dataset to teach on",
                    default="datasets/teaching")
parser.add_argument('--evaluation_prefix', type=str, help="path to folder with evaluation sub-folders",
                    default="datasets/grief_jpg")
parser.add_argument('--evaluation_paths', type=str, help="names of folders to eval",
                    default=["testing_Dec", "testing_Feb", "testing_Nov"])
parser.add_argument('--weights_folder', type=str, help="path of weights folder", default="weights/")
parser.add_argument('--weights_file', type=str, help="name of weights.pt", default="model_eunord.pt")
parser.add_argument('--weights_file2', type=str, help="name of weights.pt", default="model_tiny.pt")
parser.add_argument('--file_out', type=str, help="name of pickle out", default="_GT_.pickle")
args = parser.parse_args()

dataset_path = os.path.join(pwd, args.dataset_path)
evaluation_paths = args.evaluation_paths
evaluation_prefix = os.path.join(pwd, args.evaluation_prefix)
gt_name = "GT_merge_s_g"
GT_file = os.path.join(evaluation_prefix, gt_name + ".pickle")
init_weights = os.path.join(pwd, "init_weights.pt")
config = load_config("NN_config.yaml", 512)


def setup_missions(missions, exp_folder_name):
    for mission in missions:
        mission.setup_mission(exp_folder_name)  # setups folderrs for specific missio
        mission.plan_modifier()  # generates first plan for current strategy of the mission
        mission.setup_current_strategy()  # sets up current mission
        mission.save()


def process_new(missions, exp_folder_name):
    setup_missions(missions, exp_folder_name)
    for mission in missions:
        learning_loop(mission)


def process_old(names, exp_folder_name):
    for mission_name in names:
        mission = Mission(int(mission_name))
        mission = mission.load(os.path.join(exp_folder_name, mission_name))
        learning_loop(mission)


def learning_loop(mission, iterations=1):
    print("-----------------------------------------------------------------------")
    s_time = time.time()
    save = False
    while not mission.no_more_data:
        save = True
        print(mission.name)
        mission.c_strategy.print_parameters()
        trained = process_plan(mission)  # trains and generates new metrics
        if not trained:
            break
        grade_plan(mission)
        mission.save()
        print("Metrics: ", mission.c_strategy.next_metrics)
        mission.advance_mission(mission.c_strategy.next_metrics)
    if save:
        mission.save()
    print("Mission processing:", time.time() - s_time)


def grade_plan(mission, eval_to_file=False, grade=False):
    estimates_grade = None
    if mission.c_strategy.progress == 3 or eval_to_file:
        with open(GT_file, 'rb') as _handle:
            gt_in = pickle.load(_handle)
        estimates_grade = evaluate_for_GT(mission, evaluation_prefix,
                                          evaluation_paths,
                                          _GT=gt_in, conf=config)
        mission.c_strategy.progress = 4
    if mission.c_strategy.progress == 4 or grade:
        grade_type(mission, _GT=gt_in, estimates=estimates_grade)
        mission.c_strategy.progress = 5


def process_plan(mission, enable_teach=False, enable_eval=False, enable_metrics=True):
    start_time = time.time()
    hist_nn = None
    mission.c_strategy.file_list, count = make_combos_for_teaching(mission.c_strategy.plan, dataset_path)
    if count == 0:
        mission.c_strategy.is_faulty = True
        print("No new combos")
        return False
    if mission.c_strategy.progress == 0 or enable_teach:
        _ = teach(dataset_path, mission, init_weights=init_weights, conf=config)
        mission.c_strategy.progress = 1
        mission.c_strategy.train_time = time.time() - start_time
    if mission.c_strategy.progress == 1 or enable_eval:
        start_time2 = time.time()
        hist_nn, displacements = evaluate_for_learning(mission, dataset_path,
                                                       conf=config)
        mission.c_strategy.progress = 2
        mission.c_strategy.eval_time = time.time() - start_time2
    if mission.c_strategy.progress == 2:
        metrics = process_ev_for_training(mission, dataset_path, conf=config, hist_nn=hist_nn)
        mission.c_strategy.next_metrics = metrics
        mission.c_strategy.progress = 3
    return True


if __name__ == "__main__":
    REDO = [False, False, True, True]
    process_new(["none"], REDO)
