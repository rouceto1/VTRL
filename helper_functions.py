#def NN_eval(file_list):
import numpy as np
import copy
import os
import pickle
## adds file extension to all files in the list
def choose_proper_filetype(filetype, lst):
    file_lst = copy.deepcopy(lst).tolist()
    #print(file_lst)
    for i,f in enumerate(lst):
        #print(file_lst[i][0], f[0] + filetype, "beore ")
        file_lst[i][0] = f[0] + filetype
        file_lst[i][1] = f[1] + filetype
        #print(file_lst[i][0])
    return file_lst

def readGTFile(file_list, gt_exact_file):

    with open(gt_exact_file, 'rb') as handle:
        gt_in = pickle.load(handle) 
    gt_out = []
    for file_pair in file_list:
        path = os.path.normpath(file_pair[0])
        split_path = path.split(os.sep)
        path2 = os.path.normpath(file_pair[1])
        split_path2 = path2.split(os.sep)
        for gt_single in gt_in:
            if split_path[-1] in gt_single and split_path[-3] + "/" + split_path[-2] in gt_single:
                if split_path2[-1] in gt_single and split_path2[-3] + "/" + split_path2[-2] in gt_single:
                    gt_out.append(gt_single[0])
    ##print(gt_out)
    return gt_out


def usefull_annotation(feature_count, histogram):
    if (feature_count > 800):
        return True
    return False



## makes the file list from all images to all targets images

def make_file_list(places,targets, images,image_file_template, dataset_path,evaluation_prefix,evaluation_paths):
    combination_list2 = []
    file_list2 = []
    for e in evaluation_paths:
        for place in places: #range(7):
            for nmbr in images: #range(1,143):
                for target in targets:
                    combination_list2.append([place,target,nmbr])
        for combo in combination_list2:
            file1 = os.path.join(dataset_path, image_file_template % (combo[0],combo[1]))
            file2 = os.path.join(evaluation_prefix,e, image_file_template % (combo[0],combo[2]))
            if os.path.exists(file1 + ".png"):
                if  os.path.exists(file2 + ".png"):
                    file_list2.append([file1, file2])
    return file_list2




def make_combos_for_teaching(chosen_positions,dataset_path,image_file_template,filetype_FM):
    print("First file loaded")
    indexes = dict([(0,[]),(1,[]),(2,[]),(3,[]),(4,[]),(5,[]),(6,[]),(7,[])])
    for i, value in enumerate(chosen_positions):
        if value == -1:
            continue

        if value == 8:
            for every in range(8):
                if os.path.exists(os.path.join(dataset_path, image_file_template % (every,i)) + filetype_FM):
                    indexes[every].extend([i])
                else:
                    continue
        else:
            if os.path.exists(os.path.join(dataset_path, image_file_template % (value,i)) + filetype_FM):
                indexes[value].extend([i])

    print("indexes Made")
    ## make a combination list from all the chosen places
    combination_list = []
    all_combos = False
    added = []
    if all_combos:
        for key in indexes:
            for val in indexes[key]:
                for val2 in indexes[key]:
                    if not val == val2:
                        if not val2 in added:
                            combination_list.append([key,val,val2])
                            added.append(val)
    else:
        for key in indexes:
            for val in indexes[key]:
                if not val == indexes[key][0]:
                    combination_list.append([key,indexes[key][0],val])

    file_list = []
    for combo in combination_list:
        file1 = os.path.join(dataset_path, image_file_template % (combo[0],combo[1]))
        file2 = os.path.join(dataset_path, image_file_template % (combo[0],combo[2]))
        file_list.append([file1, file2])
    file_list= np.array(file_list)
    return file_list

