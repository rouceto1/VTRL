#!/usr/bin/env python3

from python.teach.teach import *
from python.teach.metrics import *
from python.grading.grade_results import *
import argparse
from python.helper_functions import *
import time
from python.teach.mission import Mission, Strategy
from multiprocessing import Pool, current_process
import logging
import warnings
from src.NN_new.parser_strands import get_img
from python.teach.acquisition import In_Dataset, In_Simulation

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
gt_name = "GT_new"
GT_file = os.path.join(evaluation_prefix, gt_name + ".pickle")
init_weights = os.path.join(pwd, "init_weights.pt")
config = load_config("NN_config.yaml", 512)

def setup_missions(missions, exp_folder_name):
    for mission in missions:
        mission.setup_mission(exp_folder_name)  # setups folderrs for specific missio


def process_new(generator, exp_folder_name):
    generator.save_gen(exp_folder_name, generator.get_txt())
    setup_missions(generator.missions, exp_folder_name)
    #for mission in generator.missions:
    #    learning_loop(mission)


def multi_run_wrapper(args):
    process_old(*args)

def process_old(name, cuda=None):
    conf = config.copy()
    print("-----------------------------------------------------------------------")
    print(name)
    if cuda is not None:
        cuda = current_process()._identity[0] - 1
        d = "cuda:" + str(cuda)
        print("Using cuda: ", d)
        device = t.device(d)
        conf["device"] = device
    mission = Mission(int(name))
    mission = mission.load(os.path.join(conf["exp_folder_name"], name))
    print("Simulation :  ",mission.simulation)
    learning_loop(mission, conf, simulation=mission.simulation)

def mutlithred_process_old(names, exp_folder_name, thread_limit=None):
    config["exp_folder_name"] = exp_folder_name
    if thread_limit is not None:
        data = []
        for n,name in enumerate(names):
            data.append((name, n))
        with Pool(thread_limit) as pool:
            pool.map(multi_run_wrapper, data)
            #tqdm(pool.imap(multi_run_wrapper, data), total=len(names))
    else:
        for name in names:
            process_old(name)


def learning_loop(mission, conf, cuda=None, iterations=1, simulation=False):
    s_time = time.time()
    save = False
    print(simulation)
    while not mission.no_more_data:
        save = True
        mission.c_strategy.print_parameters()
        trained = process_plan(mission, conf=conf, simulation=simulation)  # trains and generates new metrics
        if not trained:
            break
        grade_plan_vtrl(mission, conf= conf, simulation=simulation)
        mission.save()
        #print("Metrics: ", mission.c_strategy.next_metrics)
        mission.advance_mission(mission.c_strategy.ambiguity)
    if save:
        mission.save()
    #print("Mission processing:", time.time() - s_time)


#TODO - redo this for sim data
def grade_plan_vtrl(mission, eval_to_file=False, grade=False, conf=None, simulation=False):
    estimates_grade = None
    if mission.c_strategy.progress == 3 or eval_to_file:
        with open(GT_file, 'rb') as _handle:
            gt_in = pickle.load(_handle)
        estimates_grade = evaluate_for_GT(mission, evaluation_prefix,
                                          evaluation_paths,
                                          _GT=gt_in, conf=conf)
        mission.c_strategy.progress = 4
        try:
            print("Image cache 3: ", get_img.cache_info())
        except:
            pass
    if mission.c_strategy.progress == 4 or grade:
        grade_type(mission, _GT=gt_in, estimates=estimates_grade)
        mission.c_strategy.progress = 5
        try:
            print("Image cache 4: ", get_img.cache_info())
        except:
            pass

def process_plan(mission, enable_teach=False, enable_eval=False, enable_metrics=True, conf = None, simulation=False):
    start_time = time.time()
    hist_nn = None
    if simulation:
        print("Simulatiog")
        acquisition = In_Simulation("/home/rouceto1/.ros/runs")
    else :
        print("Fetching")
        acquisition = In_Dataset(dataset_path)
    mission.c_strategy.file_list, count = acquisition.make_combos_for_dataset(mission.c_strategy.timetable,
                                                                              mission.c_strategy.time_start,
                                                                              mission.c_strategy.time_limit)
    print("Used " + count)
    if count <= 1:
        mission.c_strategy.is_faulty = True
        print("No new combos")
        return False
    print("Data Ready")
    if mission.c_strategy.progress == 0 or enable_teach:
        if not mission.c_strategy.preteach or mission.c_strategy.iteration == 0:
            weights = init_weights
        else:
            weights = mission.old_strategies[-1].model_path
        _ = teach(dataset_path, mission, init_weights=weights, conf=conf)
        mission.c_strategy.progress = 1
        mission.c_strategy.train_time = time.time() - start_time
        try:
            print("Image cache 0: ", get_img.cache_info())
        except:
            pass
    if mission.c_strategy.progress == 1 or enable_eval:
        start_time2 = time.time()
        hist_nn, displacements = evaluate_for_learning(mission, dataset_path,
                                                       conf=conf)
        mission.c_strategy.progress = 2
        mission.c_strategy.eval_time = time.time() - start_time2
        try:
            print("Image cache 1: ", get_img.cache_info())
        except:
            pass
    if mission.c_strategy.progress == 2:
        ambiguity = process_ev_for_training(mission, dataset_path, conf=conf, hist_nn=hist_nn)
        mission.c_strategy.ambiguity = ambiguity
        mission.c_strategy.progress = 3
        try:
            print("Image cache 2: ", get_img.cache_info())
        except:
            pass
    return True


if __name__ == "__main__":
    pwd = os.getcwd()
    exp_path = "experiments"
    # exp_path = "backups/learning"
    experiments_path = os.path.join(pwd, exp_path)
    paths = [item for item in os.listdir(experiments_path) if os.path.isdir(os.path.join(experiments_path, item))]
    paths.sort()
    mutlithred_process_old(paths, exp_folder_name=exp_path, thread_limit=None)
