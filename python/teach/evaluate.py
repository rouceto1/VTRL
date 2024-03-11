# import src.NN_old.NNET as neuralka
import src.NN_new.evaluate_model as nn_ev
import sys
from python.helper_functions import *
import src.FM.python.python_module as grief
import cv2 as cv
import matplotlib.pyplot as plt
from python.teach.planner import VTRE
def evaluate_for_learning(mission, _dataset_path,
                          _cache2=None, conf=None):
    hist_nn, displacement_nn = NN_eval(mission.c_strategy.file_list,
                                       os.path.join(mission.experiments_path, mission.name) + "/plots",
                                       mission.c_strategy.model_path,
                                       conf)
    with open(mission.c_strategy.estimates_path, 'wb') as handle:
        pickle.dump(hist_nn, handle)

    return hist_nn, displacement_nn


def evaluate_for_GT(mission, _evaluation_prefix, _evaluation_paths, _GT=None,

                    _cache2="/tmp/cache.pkl", conf=None, simulation=False):
    if simulation:
        simulator = VTRE()
        file_list, displacements, feature_count_l, feature_count_r, matches, histograms, hist_nn= (
            simulator.simulate(mission.c_strategy.timetable, mission.c_strategy.time_start,
                               mission.c_strategy.time_end, mission.c_strategy.file_list,
                               mission.c_strategy.model_path, conf))
    else:

        file_list = (
            make_file_list_from_gt(_evaluation_prefix, _GT))
        out = [fm_nn_eval(file_list, filetype_NN, filetype_FM, mission.plot_folder, mission.c_strategy.model_path, _cache2,
                          conf=conf)]
    with open(mission.c_strategy.grading_path, 'wb') as handle:
        pickle.dump(out, handle)
    #print("GT estiamtes output at:" + str(mission.c_strategy.grading_path))
    return out


def NN_eval(file_list_nn, out_path, weights_file, conf):
    if conf["old"]:
        pass
        hist_in = None
        dsp = None
        # print("doing old")
        # a, b, hist_in, dsp = neuralka.NNeval_from_python(np.array(file_list_nn), "strands", weights_file)
    else:
        # print("doing new")
        aa, bb, hist_in, dsp = nn_ev.NNeval_from_python(np.array(file_list_nn), "strands", out_path, weights_file, conf)
    hist_in = np.float64(hist_in)
    return hist_in, dsp


"""
input: file list, filetypes for FM,NN,
weights_file for evaluation
dataset_path to evaluate
cahce_file corresponding to file_list and dataset_path that contains NN evaluation only
gt - Ground truth to pas to cpp_eval_on_file (OBSOLETE)

evaluates combos by NN and then uses this output for FM evaluation
"""


def fm_nn_eval(file_list, filetype_nn, filetype_fm, out_path, weights_file, cache_file, conf):
    count = len(file_list)
    if conf["limit"] is not None:
        count = conf["limit"]
    gt = np.zeros(count, dtype=np.float64)
    disp = np.zeros(count, dtype=np.float32)
    fcount_l = np.zeros(count, dtype=np.int32)
    fcount_r = np.zeros(count, dtype=np.int32)
    matches = np.zeros(count, dtype=np.int32)
    file_list = np.array(file_list)[:count]
    # if cache_file is not None and os.path.exists(cache_file):
    #    with open(cache_file, 'rb') as handle:
    #        hist_nn = pickle.load(handle)
    # else:
    #    pass
    hist_nn, displacement = NN_eval(choose_proper_filetype(filetype_NN, file_list),
                                    out_path, weights_file,
                                    conf)
    # if cache_file is not None:
    #    with open(cache_file, 'wb') as handle:
    #        pickle.dump(hist_nn, handle)
    hist_out = np.zeros((count, 63), dtype=np.float64)
    displacements, feature_count_l, feature_count_r, histograms = grief.cpp_eval_on_files(
        choose_proper_filetype(filetype_FM, file_list),
        disp, fcount_l, fcount_r, matches, count, hist_nn, hist_out, gt)
    # FM_out = np.array([disp, fcount], dtype=np.float64).T
    # file_list = np.array(file_list)[:count]
    np.set_printoptions(threshold=sys.maxsize)
    return file_list, displacements, feature_count_l, feature_count_r, matches, histograms, hist_nn


##
def fm_eval(file_list, filetype_FM, limit=None):
    """ Feature matcher treis to evaluate on all files in the list

    :param filetype_fm:
    :param file_list:
    :param limit:
    :return:
    """
    count = len(file_list)
    if limit is not None:
        count = limit  # THIS IS TO LIMIT IT FOR DEBUGING PURPOUSES (maybe a fnction in the future?)
    disp = np.zeros(count, dtype=np.float32)
    fcount_l = np.zeros(count, dtype=np.int32)
    fcount_r = np.zeros(count, dtype=np.int32)
    matches = np.zeros(count, dtype=np.int32)
    grief.cpp_teach_on_files(file_list, disp, fcount_l, fcount_r, matches, count)
    fm_out = np.array([disp, fcount_l, fcount_r], dtype=np.float32).T
    # file_list.append(disp)
    file_list = np.array(file_list)[:count]
    files_with_displacement = np.append(file_list, fm_out, axis=1)
    # plot_fm_list(files_with_displacement)
    return files_with_displacement
