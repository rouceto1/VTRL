import os
import numpy as np
import pandas as pd
import json
import pickle


def get_annotation_dict(path, GT_file):
    df = pd.read_csv(path)
    entries = {"stromovka": [{}], "planetarium": [{} for _ in range(11)],
               "carlevaris": [{}], "michigan": [{} for _ in range(11)],
               "cestlice_reduced": [{} for _ in range(9)]}
    additions = []
    for entry in df.iterrows():
        json_str = entry[1]["meta_info"].replace("'", "\"")
        entry_dict = json.loads(json_str)
        dataset_name = entry_dict["dataset"]
        # this is done for first against all annotation
        if "" != entry_dict["season"]:
            target_season = int(entry_dict["season"][-2:]) - 1
        else:
            continue
        img_idx = int(entry_dict["place"])
        diff = 0
        kp_dict1 = json.loads(entry[1]["kp-1"].replace("'", "\""))
        kp_dict2 = json.loads(entry[1]["kp-2"].replace("'", "\""))
        for kp1, kp2 in zip(kp_dict1, kp_dict2):
            diff += (kp1["x"] / 100) * 1024 - (kp2["x"] / 100) * 1024
        mean = diff // len(kp_dict1)
        if dataset_name == "cestlice_reduced":
            #continue
            pass
        else:
            entries[dataset_name][target_season][img_idx] = round(mean)
        shift = mean / 1024
        f_name_1 = entry_dict["dataset"] + "/" + "season_00"
        i_name_1 = "%09d" % entry_dict["place"]
        f_name_2 = entry_dict["dataset"] + "/" + entry_dict["season"]
        i_name_2 = "%09d" % entry_dict["place"]
        add = [shift, 0, 0, 0, f_name_1, i_name_1, f_name_2, i_name_2, [], [], 1]
        additions.append(add)
    return additions


def combine_annotations(list_all):
    list_all_new = []
    already_matched = []
    too_large = 0
    for entry in list_all:
        for entry2 in list_all:
            if not entry[6][:5] == entry2[6][:5]:
                continue
            if not entry[7] == entry2[7]:
                continue
            if entry == entry2:
                list_all_new.append(entry)
                continue
            combined_shift = entry2[0] - entry[0]
            if abs(combined_shift) >= 1:
                continue
            if abs(combined_shift) >= 0.5:
                too_large += 1
                continue
            if [entry, entry2] in already_matched:
                continue
            new_entry = []

            new_entry.append(combined_shift)
            new_entry.append(entry[2])
            new_entry.append(entry2[2])
            combined_matches = min(entry[3], entry2[3])
            new_entry.append(combined_matches)
            new_entry.append(entry[6])
            new_entry.append(entry[7])
            new_entry.append(entry2[6])
            new_entry.append(entry2[7])
            combined_histogram1 = entry[8] + entry2[8]
            new_entry.append(combined_histogram1)
            combined_histogram2 = entry[9] + entry2[9]
            new_entry.append(combined_histogram2)
            new_entry.append(1)
            list_all_new.append(new_entry)
            already_matched.append([entry, entry2])
    print(too_large)
    return list_all_new


def combine_GT(GT1, GT2):
    for i in GT1:
        i[8] = []
        i[9] = []
    #extend GT2 by 2000 random entries form GT1
    rng = np.random.default_rng()
    numbers = rng.choice(len(GT1), size=2000, replace=False)
    for i in numbers:
        GT2.append(GT1[i])

    return sorted(GT2, key=lambda x: x[4])

def save_GT(path,  data):
    print("saving")
    with open(path, 'wb') as handle:
        pickle.dump(data, handle)


if __name__ == "__main__":
    gt_new = get_annotation_dict("/home/rouceto1/datasets/grief_jpg/annotation.csv",
                                 "/home/rouceto1/datasets/grief_jpg/GT_redone_best.pickle")
    gt_combined = combine_annotations(gt_new)
    with open("/home/rouceto1/datasets/grief_jpg/GT_redone_best_reformat.pickle", 'rb') as handle:
        gt_in = pickle.load(handle)
    gt_out = combine_GT(gt_in, gt_combined)
    save_GT("/home/rouceto1/datasets/grief_jpg/GT_merge.pickle", gt_out)
