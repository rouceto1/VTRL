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
from helper_functions import *
from evaluate import *
#from src.FM.python.python_module import *
#from src.NN.NNET import *
print("Import functional")
#from src.NN.NNE import *

dataset_path = "/home/rouceto1/datasets/strands_crop/training_Nov"
evaluation_prefix = "/home/rouceto1/datasets/strands_crop"
evaluation_paths = ["testing_Dec", "testing_Feb", "testing_Nov"]

GT_file = "gt.pkl"
chosen_positions_file = "input.txt"
weights_file = "exploration.pt"
feature_matcher_file = "FM_out.pickle"
neural_network_file = "NN_out.pickle"
annotation_weights = "GOOOD.pt"

cache = os.path.join(dataset_path, feature_matcher_file )
cache2 = os.path.join(dataset_path, neural_network_file )
filetype_FM = ".bmp"
filetype_NN = ".png"
image_file_template = "place_%d/%05d"
use_cache = False

chosen_positions = np.loadtxt(os.path.join(dataset_path, chosen_positions_file),int)


def teach():
    if not os.path.exists(cache) or not use_cache:
        file_list = make_combos_for_teaching(chosen_positions,
                                             dataset_path,
                                             image_file_template,
                                             filetype_FM)

        print("evaling FM")
        files_with_displacement = FM_eval(file_list,filetype_FM)
        with open(cache, 'wb') as handle:
            pickle.dump(files_with_displacement,
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("making new chace " + cache)
    else:
        print("reading cache " + cache)
        with open(cache, 'rb') as handle:
            files_with_displacement = pickle.load(handle)
    print("Displcaments aquired " + cache)
    #print(files_with_displacement)
    ## teach NN on all the combinsations
    #print (choose_proper_filetype(filetype_NN, files_with_displacement))
    neuralka.NNteach_from_python(np.array(choose_proper_filetype(filetype_NN,files_with_displacement)),
                                 "strands",
                                 os.path.join(dataset_path,
                                              weights_file),
                                 3)

## EVAL->
## what files to eval
def evaluate():
    #make file list agaisnt first images (original map)
    file_list = make_file_list(range(7), [0],range(1,143), image_file_template,dataset_path,evaluation_prefix,evaluation_paths)

    gt = readGTFile(file_list,os.path.join(evaluation_prefix, GT_file ))

    displacements,feature_count,histograms =  FM_NN_eval(file_list,filetype_NN,filetype_FM,weights_file,dataset_path,cache2,use_cache,gt)
    return file_list, displacements,feature_count,histograms


def readUndecidedGTFile():
    pass

def annotate():
    file_list = make_file_list(range(7), [0],range(1,143), image_file_template,dataset_path,evaluation_prefix,evaluation_paths)

    gt = np.zeros(10, dtype = np.float64)
    displacements,feature_count,histograms =  FM_NN_eval(file_list,filetype_NN,filetype_FM,weights_file,dataset_path,cache2,use_cache,gt)

    annotations = []
    for i in range(len(displacements)):
        if usefull_annotation(feature_count[i], histograms[i]):
            out = [displacements[i]]
        else:
            out = [float('nan')]
        out.append(feature_count[i])
        path = os.path.normpath(file_list[i][0])
        split_path = path.split(os.sep)
        out.append(split_path[-3] + "/" + split_path[-2])
        out.append(split_path[-1])
        path2 = os.path.normpath(file_list[i][1])
        split_path2 = path2.split(os.sep)
        out.append(split_path2[-3] + "/" + split_path2[-2])
        out.append(split_path2[-1])
        out.append(histograms[i])
        annotations.append(out)

    gt_out = os.path.join(evaluation_prefix, GT_file)
    with open(gt_out, 'wb') as handle:
        pickle.dump(annotations, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("GT written " + str(gt_out))
    print(annotations)


if __name__ == "__main__":
    annotate()
    teach()
    evaluate()

