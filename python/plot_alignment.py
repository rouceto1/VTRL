#!/usr/bin/env python3

import ctypes
import sys
import json
import os
import copy
import numpy as np
import pickle

from matplotlib import pyplot as plt



dataset_path = "/home/rouceto1/datasets/strands_crop/training_Nov"
weights_file = "exploration.pt"
eval_out_file = weights_file + "_eval.pickle"
eval_out = os.path.join(dataset_path,eval_out_file)

def filter_to_max(lst, threshold):
    lst[lst>threshold] = threshold
    lst[lst<-threshold] = -threshold
    return lst


'''
data in in format [
file_list[
fileA, fileB
] these are missing file extensions. can be both png and bmp

displcement[ in pixels from A to B]

feature_count[ in num featrres TODO i think detected]

histogram_from_FM[ [features for each bin] ]

histogram_from_NN[ [likliehood distribution rather then actual histogram] ]

]
'''
def plot_alignments(data):
    file_list = data[0][0]
    disp = data[0][1]
    feature_count = data[0][2]
    histogram_FM = data[0][3]
    histogram_NN = data[0][4]
    print (len(disp))
    disp = filter_to_max(disp,300)
    plt.plot(disp)
    plt.show()


if __name__ == "__main__":
    with open(eval_out, 'rb') as handle:
        things_out = pickle.load(handle)
    plot_alignments(things_out)
