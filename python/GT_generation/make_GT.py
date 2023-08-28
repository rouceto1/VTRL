import os
import numpy as np
import pandas as pd
import json
import pickle
import manual_image_alignment


def get_annotation_dict(path):
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
            # continue
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
    # extend GT2 by 2000 random entries form GT1
    rng = np.random.default_rng()
    numbers = rng.choice(len(GT1), size=2000, replace=False)
    print("The divide is at: ", len(GT2), " minus all other lines: ")
    for i in numbers:
        GT2.append(GT1[i])

    return sorted(GT2, key=lambda x: x[4])


def save_GT(path, data):
    print("saving")
    with open(path, 'wb') as handle:
        pickle.dump(data, handle)


def sanity_check(GT):
    combos = []
    faults = []
    for i in GT:
        if abs(i[0]) > 0.5:
            print("too large GT", i)
            faults.append(i)
        if i[4] + i[5] + i[6] + i[7] in combos:
            print("double entry", i)
            faults.append(i)
        combos.append(i[4] + i[5] + i[6] + i[7])

    for f in faults:
        GT.remove(f)
    return GT


if __name__ == "__main__":
    dataset_path = os.path.join("/home/rouceto1/datasets/grief_jpg")
    original_strands_annotations = os.path.join(dataset_path, "GT_redone.pickle")
    combined_strands_annotations = os.path.join(dataset_path, "GT_redone_best.pickle")
    original_grief_annotations = os.path.join(dataset_path, "annotation.csv")
    final_GT_path = os.path.join(dataset_path, "GT_merge.pickle")

    annotation_grief = manual_image_alignment.Annotate(original_strands_annotations, combined_strands_annotations,
                                                       dataset_path)
    # annotation.annotate()
    annotation_grief.combine_annotations()
    # annotation.show_GT()
    with open(combined_strands_annotations, 'rb') as handle:
        strands_gt_combined = pickle.load(handle)

    grief_gt_basic = get_annotation_dict(original_grief_annotations)
    grief_gt_combined = combine_annotations(grief_gt_basic)

    gt_out = combine_GT(strands_gt_combined, grief_gt_combined)
    gt_out = combine_GT(strands_gt_combined, [])
    gt_chacked = sanity_check(gt_out)
    save_GT(final_GT_path, gt_chacked)
