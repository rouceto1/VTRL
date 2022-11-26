#!/usr/bin/env python3

import ctypes
import sys
import json
import os
import copy
import numpy as np
import pickle

from matplotlib import image
from matplotlib import pyplot as plt

annotation_file = "1-2-fast-grief.pt"
GT_file = annotation_file + "_GT_.pickle"
evaluation_prefix = "/home/rouceto1/datasets/strands_crop"
gt_file_in = os.path.join(evaluation_prefix, GT_file)


dataset_path = "/home/rouceto1/datasets/strands_crop/training_Nov"
weights_file = "exploration.pt"
eval_out_file = weights_file + "_eval.pickle"
eval_out = os.path.join(dataset_path,eval_out_file)

def filter_to_max(lst, threshold):
    lst[lst>threshold] = threshold
    lst[lst<-threshold] = -threshold
    return lst

def add_img_to_plot(plot, img_path):
    img = image.imread(img_path+ ".bmp")
    #plot.imshow(img)
    plot.imshow(img,aspect="auto")


def read_GT(gt):
    gt_disp = []
    gt_features = []
    gt_placeA = []
    gt_timeA = []
    gt_placeB = []
    gt_timeB = []
    gt_histogram_FM = []
    gt_histogram_NN = []
    for place in gt:
        gt_disp.append(place[0])
        #gt_features.append(place[1])
        gt_placeA.append(place[2] + "/" + place[3])
        #gt_timeA.append(place[3])
        gt_placeB.append(place[4] + "/" + place[5])
        #gt_timeB.append(place[5])
        #gt_histogram_FM.append(place[6])
        #gt_histogram_NN.append(place[7])
    return gt_disp, gt_placeA,gt_placeB



##Function to match evaluated things to ground truth, for now it is ordered TODO
def match_gt_to_eval(gt_placeA,gt_placeB,file_list,gt_disp):
    return gt_disp



def compute_with_plot(data,gt):
    gt_disp, gt_placeA,gt_placeB = read_GT(gt)
    file_list = data[0][0]
    histogram_FM = data[0][3]
    histogram_NN = data[0][4]
    feature_count = data[0][2]
    displacement = data[0][1]
    gt_disp = match_gt_to_eval(gt_placeA,gt_placeB,file_list,gt_disp)
    disp, line, line_integral = compute(displacement,gt_disp)
    plt.plot(line[0],line[1])
    plt.show()
    for location in range(50,len(file_list)):
        f, axarr = plt.subplots(3)
        for i in [0,1]:
            add_img_to_plot(axarr[i],file_list[location][i])
        #print(file_list[location])
        r1 =range(-630,630,1260//63)
        r2 = range(-504,504,1008//63)
        axarr[2].axvline(x=gt_disp[location], ymin=0, ymax=1, c="b", ls="--")
        axarr[2].axvline(x=displacement[location], ymin=0, ymax=1, c="k", ls="--")
        axarr[2].plot(r1,histogram_FM[location]/max(histogram_FM[location]),c="r")
        axarr[2].plot(r1,histogram_NN[location]/max(histogram_NN[location]),c="g")
        axarr[2].legend(["GT", "displ", "h_FM","h_NN"])
        
        f.tight_layout()
        plt.show()
        plt.savefig("./comparison/" + file_list[location][1][-5:] + ".png")
        plt.close()

    return(line,line_integral)

def compute(displacement,gt):
    disp = displacement - gt
    line = compute_AC_curve(filter_to_max(disp,500))
    line_integral = get_integral_from_AC(line)

    return disp, line,line_integral

def compute_AC_curve(error):
    disp = np.sort(abs(error))
    length = len(error)
    return [disp, np.array(range(length))/length]


## TODO this is probably incorect since it just summs all the errors therefore not normalised
def get_integral_from_AC(AC):
    suma = 0
    for val in AC[0]:
        if val == 500:
            break
        suma = suma + val
    return print(suma)



if __name__ == "__main__":
    print("loading")
    with open(eval_out, 'rb') as handle:
        things_out = pickle.load(handle)
    print("loaded eval data")
    with open(gt_file_in, 'rb') as handle:
        gt_out = pickle.load(handle)
    print("loaded GT")
    compute_with_plot(things_out,gt_out)
