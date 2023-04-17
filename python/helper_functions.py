import numpy as np
import copy
import os
import pickle as pickle
import yaml
import torch as t
from pathlib import Path
filetype_FM = ".bmp"
filetype_NN = ".png"
image_file_template = "place_%d/%05d"
chosen_positions_file = "input.txt"
feature_matcher_file = "FM_out.pickle"


# adds file extension to all files in the list
def choose_proper_filetype(filetype, lst):
    file_lst = copy.deepcopy(lst).tolist()
    for i, f in enumerate(lst):
        file_lst[i][0] = f[0] + filetype
        file_lst[i][1] = f[1] + filetype
    return file_lst


def read_gt_file(file_list, gt_exact_file):
    with open(gt_exact_file, 'rb') as handle:
        gt_in = pickle.load(handle)
    gt_out = []
    test = 0
    for file_pair in file_list:
        test = test + 1
        path = os.path.normpath(file_pair[0])
        split_path = path.split(os.sep)
        path2 = os.path.normpath(file_pair[1])
        split_path2 = path2.split(os.sep)
        time = split_path[-1]
        place = split_path[-3] + "/" + split_path[-2]
        time2 = split_path2[-1]
        place2 = split_path2[-3] + "/" + split_path2[-2]
        for gt_single in gt_in:
            second = place2 in gt_single and time2 in gt_single
            if second:
                first = place in gt_single and time in gt_single
                if first:
                    # SOLVED: this adds more things than it shoudl. Prolly bad variable --SOLVED-- was isse with file list making
                    gt_out.append(gt_single[0])
    return gt_out


def usefull_annotation(feature_count, histogram):
    if feature_count > 800:
        return True
    return False


# makes the file list from all images to all targets images

def make_file_list(places, targets, images, dataset_path, evaluation_prefix, evaluation_paths):
    combination_list2 = []
    file_list2 = []
    for place in places:  # range(7):
        for nmbr in images:  # range(1,143):
            for target in targets:
                combination_list2.append([place, target, nmbr])
    for e in evaluation_paths:
        for combo in combination_list2:
            file1 = os.path.join(dataset_path, image_file_template % (combo[0], combo[1]))
            file2 = os.path.join(evaluation_prefix, e, image_file_template % (combo[0], combo[2]))
            if os.path.exists(file1 + ".png") and os.path.exists(file2 + ".png"):
                file_list2.append([file1, file2])
    return file_list2


def make_combos_for_teaching(chosen_positions, dataset_path, filetype_fm, all_combos=False, limit=None):
    #print("First file loaded")
    indexes = dict([(0, []), (1, []), (2, []), (3, []), (4, []), (5, []), (6, []), (7, [])])
    for i, value in enumerate(chosen_positions):
        if limit == i:
            break
        if value == -1:
            continue

        if value == 8:
            for every in range(8):
                if os.path.exists(os.path.join(dataset_path, image_file_template % (every, i)) + filetype_fm):
                    indexes[every].extend([i])
                else:
                    continue
        else:
            if os.path.exists(os.path.join(dataset_path, image_file_template % (value, i)) + filetype_fm):
                indexes[value].extend([i])

    #print("indexes Made")
    # make a combination list from all the chosen places
    combination_list = []
    added = []
    if all_combos:
        for key in indexes:
            for val in indexes[key]:
                for val2 in indexes[key]:
                    if val != val2:
                        if val2 not in added:
                            combination_list.append([key, val, val2])
                            added.append(val)
    else:
        for key in indexes:
            for val in indexes[key]:
                if not val == indexes[key][0]:
                    combination_list.append([key, indexes[key][0], val])

    file_list = []
    for combo in combination_list:
        file1 = os.path.join(dataset_path, image_file_template % (combo[0], combo[1]))
        file2 = os.path.join(dataset_path, image_file_template % (combo[0], combo[2]))
        file_list.append([file1, file2])
    file_list = np.array(file_list)
    return file_list

def get_pad(crop):
    return (crop - 8) // 16

def load_config(conf_path, image_width=512, image_height=384):
    conf = yaml.safe_load(Path(conf_path).read_text())
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
    conf["device"] = device
    output_size = conf["width"] // conf["fraction"]
    PAD = get_pad(conf["crop_sizes"][0])
    conf["pad"] = PAD
    conf["output_size"] = output_size
    conf["crop_size_eval"] = conf["width"] - 8
    conf["crop_size_teach"] = conf["crop_sizes"][0]
    conf["pad_eval"] = get_pad(conf["crop_size_eval"])
    conf["pad_teach"] = get_pad(conf["crop_size_teach"])
    conf["batching"] = conf["crops_multiplier"]
    conf["size_frac"] = conf["width"] / image_width
    conf["image_height"] = image_height
    return conf
