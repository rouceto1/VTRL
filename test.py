#!/usr/bin/env python3

import ctypes
import sys
import json
import os
import copy
import numpy as np
import pickle
import src.FM.python.python_module as grief
import src.NN.NNET as neuralka
#from src.FM.python.python_module import *
#from src.NN.NNET import *
print("Import functional")
#from src.NN.NNE import *

dataset_path = "/home/rouceto1/datasets/strands_crop/training_Nov"
evaluation_prefix = "/home/rouceto1/datasets/strands_crop"
evaluation_paths = ["testing_Dec", "testing_Feb", "testing_Nov"]

GT_file = "gt.txt"
chosen_positions_file = "input.txt"
weights_file = "exploration.pt"
feature_matcher_file = "FM_out.pickle"
neural_network_file = "NN_out.pickle"

cache = os.path.join(dataset_path, feature_matcher_file )
cache2 = os.path.join(dataset_path, neural_network_file )
filetype_FM = ".bmp"
filetype_NN = ".png"
image_file_template = "place_%d/%05d"
use_cache = True

chosen_positions = np.loadtxt(os.path.join(dataset_path, chosen_positions_file),int)

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



def FM_evla_files(file_list, filetype):
    count = len(file_list)
    count = 50 ## THIS IS TO LIMIT IT FOR DEBUGING PURPOUSES (may be a fnction in the future?)
    disp = np.zeros(count, dtype = np.float32)
    fcount = np.zeros(count, dtype = np.int32)

## Feature matcher treis to evaluate on all files in the list 
def FM_eval (file_list):
    count = len(file_list)
    count = 50 ## THIS IS TO LIMIT IT FOR DEBUGING PURPOUSES (may be a fnction in the future?)
    disp = np.zeros(count, dtype = np.float32)
    fcount = np.zeros(count, dtype = np.int32)

    grief.cpp_teach_on_files(choose_proper_filetype(filetype_FM,file_list), disp, fcount, count)
    FM_out = np.array([disp, fcount], dtype=np.float32).T
    #file_list.append(disp)
    file_list = np.array(file_list)[:count]
    files_with_displacement = np.append(file_list, FM_out, axis=1)
    return files_with_displacement

def FM_NN_eval(file_list):
    count = len(file_list)
    count = 50
    disp = np.zeros(count, dtype = np.float32)
    fcount = np.zeros(count, dtype = np.int32)
    gt = np.zeros(count, dtype = np.float64)
    file_list = np.array(file_list)[:count]


    if not os.path.exists(cache2) or not use_cache:
        a, b, hist_in, dsp = neuralka.NNeval_from_python(np.array(choose_proper_filetype(filetype_NN, file_list)),"strands", os.path.join(dataset_path, weights_file))
        hist_in = np.float64(hist_in)
        with open(cache2, 'wb') as handle:
            pickle.dump(hist_in, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("making chache" + str(cache2))
    else:
        print("reading cache" + str(cache2))
        with open(cache2, 'rb') as handle:
            hist_in = pickle.load(handle)


    hist_out = np.zeros((count,63), dtype = np.float64)
    #print(hist_in.shape)
    #print(hist_out.shape)
    d,fc,hist = grief.cpp_eval_on_files(choose_proper_filetype(filetype_FM,file_list), disp, fcount, count, hist_in, hist_out, gt)
    FM_out = np.array([disp, fcount], dtype=np.float64).T
    file_list = np.array(file_list)[:count]
    np.set_printoptions(threshold=sys.maxsize)
    #NN_eval()

def make_combos_for_teaching():
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
    print(file_list)
    ## call FM on all combinations
    print("evaling FM")
    files_with_displacement = FM_eval(file_list)
    with open(cache, 'wb') as handle:
        pickle.dump(files_with_displacement, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return files_with_displacement
def teach():
    if not os.path.exists(cache) or not use_cache:
        files_with_displacement = make_combos_for_teaching()
        print("making new chace " + cache)
    else:
        print("reading cache " + cache)
        with open(cache, 'rb') as handle:
            files_with_displacement = pickle.load(handle)
            
    print("Displcaments aquired " + cache)
    #print(files_with_displacement)
    ## teach NN on all the combinsations
    #print (choose_proper_filetype(filetype_NN, files_with_displacement))
    neuralka.NNteach_from_python(np.array(choose_proper_filetype(filetype_NN, files_with_displacement)),"strands", os.path.join(dataset_path, weights_file),3 )

## EVAL->
## what files to eval
def evaluate():
    combination_list2 = []
    file_list2 = []
    for e in evaluation_paths:
        for places in range(7):
            for nmbr in range(1,143):
                combination_list2.append([places,0,nmbr])
        for combo in combination_list2:
            file1 = os.path.join(dataset_path, image_file_template % (combo[0],combo[1]))
            file2 = os.path.join(evaluation_prefix,e, image_file_template % (combo[0],combo[2]))
            if os.path.exists(file1 + ".png"):
                if  os.path.exists(file2 + ".png"):
                    file_list2.append([file1, file2])

    FM_NN_eval(file_list2)

if __name__ == "__main__":
   # teach()
    evaluate()

