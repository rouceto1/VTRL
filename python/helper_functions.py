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
    gt.columns=["displacement", "A_date", "A_time", "B_date", "B_time"]
    #gt["A_time"] = gt.A_time + ".png"
    #gt["B_time"] = gt.B_time + ".png"
    gt[["A_month", "A_place"]] = gt.A_date.str.split("/", expand=True)
    gt[["B_month", "B_place"]] = gt.B_date.str.split("/", expand=True)
    new_ind = [f"{pic}_{key}" for pic in ["A", "B"] for key in ["month", "place", "time"]]
    for c in new_ind:
        gt[c] = gt[c].astype("category")
        gt[c] = gt[c].cat.as_ordered()
    gt = gt.set_index(new_ind)

    # print(files.join(gt, how="inner").displacement)

    return files.join(gt, how="inner")["displacement"].values

def read_gt_file_old(file_list, gt_in):
    gt_out = []
    for file_pair in tqdm(file_list):
        split_path = os.path.normpath(file_pair[0]).split(os.sep)
        split_path2 = os.path.normpath(file_pair[1]).split(os.sep)
        time = split_path[-1]
        place = split_path[-3] + "/" + split_path[-2]
        time2 = split_path2[-1]
        place2 = split_path2[-3] + "/" + split_path2[-2]
        for gt_single in gt_in:
            second = (place2 == gt_single[4] and time2 == gt_single[5]) or (place2 == gt_single[6] and time2 == gt_single[7])
            if second:
                if (place == gt_single[4] and time == gt_single[5]) :
                    gt_out.append(gt_single[0])  # TODO this might be flipped
                    break
                elif (place == gt_single[6] and time == gt_single[7]):
                    gt_out.append(-gt_single[0])
                    break
    return gt_out


def usefull_annotation(feature_count_l,feature_count_r, matches, histogram):
    if matches > 5:
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


def make_file_list_annotation(places, images, evaluation_prefix, evaluation_paths):
    out = []
    count = 0
    for place in places:
        files = []
        for subfolder in evaluation_paths:
            for time in images:
                file = os.path.join(evaluation_prefix, subfolder, image_file_template % (place, time))
                if os.path.exists(
                        os.path.join(evaluation_prefix, subfolder, image_file_template % (place, time)) + ".png"):
                    files.append(file)
                    count+=1
        combinations = list(it.combinations(files, 2))
        out.extend(combinations)
    return out

def make_file_list_from_gt(evaluation_prefix,gt):
    file_list = []
    for i in range(10000):
        sample_list = random.choices(gt, k=2)
        set = []
        set.append(os.path.join(evaluation_prefix, sample_list[4],sample_list[5]))
        set.append(os.path.join(evaluation_prefix, sample_list[6], sample_list[7]))
        file_list.append(set)
    return file_list


def make_combos_for_teaching(chosen_positions, dataset_path, filetype_fm, conf=None):
    # print("First file loaded")
    if conf["limit"] is not None:
        limit = conf["limit"]*10
    else:
        limit = -1
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

    # print("indexes Made")
    # make a combination list from all the chosen places
    combination_list = []
    cout = 0
    file_list = []
    if conf["all_combos"] is True:
        for key in indexes:
            position = list(itertools.combinations(indexes[key],2))
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
