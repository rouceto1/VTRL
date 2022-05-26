#!/usr/bin/env python3

import ctypes
import sys
import os
import numpy as np
import pickle
from src.FM.python.python_module import *
from src.NN.NNT import *
print("Import functional")

dataset_path = "/home/rouceto1/datasets/eplore/training_Nov"
chosen_positions_file = "input.txt"
weights_file = "exploration.pt"
feature_matcher_file = "FM_out.pickle"
neural_network_dile = "NN_out.pickle"

cache = os.path.join(dataset_path, feature_matcher_file )

image_file_template = "place_%d/%05d.png"


chosen_positions = np.loadtxt(os.path.join(dataset_path, chosen_positions_file),int)

def FM_eval (file_list):
    count = len(file_list)
    count = 50 ## THIS IS TO LIMIT IT FOR DEBUGING PURPOUSES (may be a fnction in the future?)
    disp = np.zeros(count, dtype = np.float32)
    fcount = np.zeros(count, dtype = np.int32)
    cpp_teach_on_files(file_list, disp, fcount, count)
    FM_out = np.array([disp, fcount], dtype=np.float32).T
    #file_list.append(disp)
    file_list = np.array(file_list)[:count]
    files_with_displacement = np.append(file_list, FM_out, axis=1)
    return files_with_displacement

print("First file loaded")
indexes = dict([(0,[]),(1,[]),(2,[]),(3,[]),(4,[]),(5,[]),(6,[]),(7,[])])
for i, value in enumerate(chosen_positions):
    if value == -1:
        continue

    if value == 8:
        for every in range(8):
            if os.path.exists(os.path.join(dataset_path, image_file_template % (every,i))):
                indexes[every].extend([i])
            else:
                continue
    else:
        if os.path.exists(os.path.join(dataset_path, image_file_template % (value,i))):
            indexes[value].extend([i])
#print(indexes)

print("indexes Made")
## make a combination list from all the chosen places
combination_list = []
for key in indexes:
    for val in indexes[key]:
        if val is not indexes[key][0]:
            combination_list.append([key,indexes[key][0],val])
            #print((np.array(combination_list))) ## combination list has place | imga | imgb

file_list = []
for combo in combination_list:
    file1 = os.path.join(dataset_path, image_file_template % (combo[0],combo[1]))
    file2 = os.path.join(dataset_path, image_file_template % (combo[0],combo[2]))
    file_list.append([file1, file2])
#print(file_list)
## call FM on all combinations

if not os.path.exists(cache) :
    files_with_displacement = FM_eval(file_list)
    with open(cache, 'wb') as handle:
        pickle.dump(files_with_displacement, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(cache, 'rb') as handle:
        files_with_displacement = pickle.load(handle)

print("Displcaments aquired")
#print(files_with_displacement)
## teach NN on all the combinsations

NNteach_from_python(files_with_displacement,"strands", os.path.join(dataset_path, weights_file) )




