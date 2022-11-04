#!/usr/bin/env python3

import ctypes
import sys
import json
import os
import copy
import numpy as np
import pickle




dataset_path = "/home/rouceto1/datasets/strands_crop/training_Nov"
weights_file = "exploration.pt"
eval_out_file = weights_file + "_eval.pickle"
eval_out = os.path.join(dataset_path,eval_out_file)

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
    

    print(data)



if __name__ == "__main__":
    with open(eval_out, 'rb') as handle:
        things_out = pickle.load(handle)
    plot_alignments(things_out)
