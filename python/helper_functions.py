import itertools

import numpy as np
import copy
import os
import pandas as pd
import pickle as pickle
import yaml
import torch as t
import itertools as it
from pathlib import Path
from tqdm import tqdm
import random
filetype_FM = ".bmp"
filetype_NN = ".png"
image_file_template = "place_%02d/%05d" #TODO redo all datasets to this format


# adds file extension to all files in the list
def choose_proper_filetype(filetype, lst):
    file_lst = copy.deepcopy(lst).tolist()
    for i, f in enumerate(lst):
        file_lst[i][0] = f[0] + filetype
        file_lst[i][1] = f[1] + filetype
    return file_lst


def read_gt_file(file_list, gt_in):
    files = pd.DataFrame.from_records(file_list, columns=["A", "B"])
    files[["A_month", "A_place", "A_time"]] = files.A.str.split("/", expand=True).iloc[:, -3:]
    files[["B_month", "B_place", "B_time"]] = files.B.str.split("/", expand=True).iloc[:, -3:]
    files.drop(["A", "B"], axis="columns", inplace=True)
    for c in files.columns:
        files[c] = files[c].astype("category")
        files[c] = files[c].cat.as_ordered()
    files = files.reset_index()
    files = files.set_index(list(files.columns[1:]))
    gt = pd.DataFrame.from_records(gt_in)
    gt = gt.iloc[:, [0, 4, 5, 6, 7]]
    gt.columns = ["displacement", "A_date", "A_time", "B_date", "B_time"]
    # gt["A_time"] = gt.A_time + ".png"
    # gt["B_time"] = gt.B_time + ".png"
    gt[["A_month", "A_place"]] = gt.A_date.str.split("/", expand=True)
    gt[["B_month", "B_place"]] = gt.B_date.str.split("/", expand=True)
    new_ind = [f"{pic}_{key}" for pic in ["A", "B"] for key in ["month", "place", "time"]]
    for c in new_ind:
        gt[c] = gt[c].astype("category")
        gt[c] = gt[c].cat.as_ordered()
    gt = gt.set_index(new_ind)
    return files.join(gt, how="inner")["displacement"].values


def usefull_annotation(feature_count_l, feature_count_r, matches, histogram):
    # 5 matches is not really goot, totaol fature count is better indicator since in dark it just does find some matches anyway
    if matches > 5:
        return True
    return False


# makes the file list from all images to all targets images
def make_file_list_annotation(places, images, evaluation_prefix, evaluation_paths, target=112):
    combination_list2 = []
    file_list2 = []
    for place in places:  # range(7):
        for nmbr in images:  # range(1,143):
            combination_list2.append([place, target, nmbr])
    for e in evaluation_paths:
        for combo in combination_list2:
            file1 = os.path.join(evaluation_prefix, evaluation_paths[0], image_file_template % (combo[0], combo[1]))
            file2 = os.path.join(evaluation_prefix, e, image_file_template % (combo[0], combo[2]))
            if os.path.exists(file1 + ".png") and os.path.exists(file2 + ".png"):
                file_list2.append([file1, file2])
    return file_list2


# this should be same for all datasets......
def make_file_list_from_gt(evaluation_prefix, gt):
    file_list = []
    use_all = False
    if use_all:
        for i in range(10000):
            sample_list = random.choices(gt)[0]
            while (useful_GT(sample_list[1], sample_list[2], sample_list[3], sample_list[10]) == False):
                sample_list = random.choices(gt)[0]
            set = []
            set.append(os.path.join(evaluation_prefix, sample_list[4], sample_list[5]))
            set.append(os.path.join(evaluation_prefix, sample_list[6], sample_list[7]))
            file_list.append(set)
    else:
        for i in gt:
            if (useful_GT(i[1], i[2], i[3], i[10]) == False):
                continue
            set = []
            set.append(os.path.join(evaluation_prefix, i[4], i[5]))
            set.append(os.path.join(evaluation_prefix, i[6], i[7]))
            file_list.append(set)

    return file_list


def useful_GT(feature_count_l, feature_count_r, matches, usefulness):
    if usefulness == 1:
        return True
    return False

def concatenate_combos_for_teaching(list1,list2):
    out = []
    out.append(list1)
    out.extend(list2)
    return out

def make_combos_for_teaching(chosen_positions, dataset_path, filetype_fm, conf=None):
    # print("First file loaded")
    #takes list of sorted chosen positions indicated by the number bigger than or equal zero and makes a combination of all indexies for a given postitions,
    # if -1 is in chosen position it skips this index
    # if -2 is chosen position it uses all possible positions for this index
    indexes = dict([(0, []), (1, []), (2, []), (3, []), (4, []), (5, []), (6, []), (7, [])])
    for i, value in enumerate(chosen_positions):
        if value == -1:
            continue

        if value == -2:
            for every in range(8):
                if os.path.exists(os.path.join(dataset_path, image_file_template % (every, i)) + filetype_fm):
                    indexes[every].extend([i])
                else:
                    continue
        else:
            if os.path.exists(os.path.join(dataset_path, image_file_template % (value, i)) + filetype_fm):
                indexes[value].extend([i])

    # print("indexes Made")
    # make a combination list from all the chosen places
    combination_list = []
    cout = 0
    file_list = []
    if conf["all_combos"] is True:
        for key in indexes:
            position = list(itertools.combinations(indexes[key], 2))
            combination_list.append(position)
            for pose in position:
                cout += 1
                file1 = os.path.join(dataset_path, image_file_template % (key, pose[0]))
                file2 = os.path.join(dataset_path, image_file_template % (key, pose[1]))
                file_list.append([file1, file2])
    else:
        for key in indexes:
            for val in indexes[key]:
                if not val == indexes[key][0]:
                    combination_list.append([key, indexes[key][0], val])
        for combo in combination_list:
            file1 = os.path.join(dataset_path, image_file_template % (combo[0], combo[1]))
            file2 = os.path.join(dataset_path, image_file_template % (combo[0], combo[2]))
            file_list.append([file1, file2])

    file_list = np.array(file_list)
    print("In total to teach on: " + str(cout))
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
