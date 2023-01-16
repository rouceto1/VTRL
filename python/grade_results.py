#!/usr/bin/env python3
import os
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
eval_out = os.path.join(dataset_path, eval_out_file)


def filter_to_max(lst, threshold):
    lst[lst > threshold] = threshold
    lst[lst < -threshold] = -threshold
    return lst


def add_img_to_plot(plot, img_path):
    img = image.imread(img_path + ".bmp")
    # plot.imshow(img)
    plot.imshow(img, aspect="auto")


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
        # gt_features.append(place[1])
        gt_placeA.append(place[2] + "/" + place[3])
        # gt_timeA.append(place[3])
        gt_placeB.append(place[4] + "/" + place[5])
        # gt_timeB.append(place[5])
        # gt_histogram_FM.append(place[6])
        # gt_histogram_NN.append(place[7])
    return gt_disp, gt_placeA, gt_placeB


##Function to match evaluated things to ground truth, for now it is ordered TODO
def match_gt_to_eval(gt_placeA, gt_placeB, file_list, gt_disp):
    return gt_disp


def load_data(data, gt):
    gt_disp, gt_placeA, gt_placeB = read_GT(gt)
    file_list = data[0][0]
    histogram_FM = data[0][3]
    histogram_NN = data[0][4]
    feature_count = data[0][2]
    displacement = data[0][1]
    gt_disp = match_gt_to_eval(gt_placeA, gt_placeB, file_list, gt_disp)
    return file_list, histogram_FM, histogram_NN, feature_count, displacement, np.array(gt_disp)


##TODO do this function
def get_streak(disp):
    streak = []
    for i in range(500):
        length = 0
        temp = 0
        for dis in disp:
            if dis < i:
                temp += 1
            else:
                if temp > length:
                    length = temp
                temp = 0
        if temp > length:
            length = temp
        streak.append(length)
    plt.plot(streak)
    plt.show()
    return streak


def compute_to_file(data, gt, dist):
    line_out = os.path.join(dist, "line.pkl")
    streak_out = os.path.join(dist, "streak.pkl")
    file_list, histogram_FM, histogram_NN, feature_count, displacement, gt_disp = load_data(data, gt)
    disp, line, line_integral, streak = compute(displacement, gt_disp)
    with open(line_out, 'wb') as handle:
        pickle.dump(line, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Line written " + str(line_out))
    with open(streak_out, 'wb') as handle:
        pickle.dump(streak, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("streak " + str(streak_out))
    out = []
    out.append(w)
    out.append(weights_file)
    out.append(line_integral)
    out.append(streak_integral)
    out.append(get_percentile(line, 50))
    out.append(get_percentile(line, 90))
    out.append(get_percentile(line, 99))


def compute_with_plot(data, gt):
    file_list, histogram_FM, histogram_NN, feature_count, displacement, gt_disp = load_data(data, gt)
    disp, line, line_integral, streak = compute(displacement, gt_disp)
    print("Results (disp,line,line_integral):")
    print(disp)
    print(line)
    print(line_integral)
    plt.plot(line[0], line[1])
    plt.grid()
    plt.show()
    for location in range(50, len(file_list)):
        f, axarr = plt.subplots(3)
        for i in [0, 1]:
            add_img_to_plot(axarr[i], file_list[location][i])
        # print(file_list[location])
        r1 = range(-630, 630, 1260 // 63)  ##for FM since it is using full imagees
        r2 = range(-504, 504, 1008 // 63)  ##for NN since it is using reduced images
        axarr[2].axvline(x=gt_disp[location], ymin=0, ymax=1, c="b", ls="--")
        axarr[2].axvline(x=displacement[location], ymin=0, ymax=1, c="k", ls="--")
        axarr[2].plot(r1, histogram_FM[location] / max(histogram_FM[location]), c="r")
        axarr[2].plot(r1, histogram_NN[location] / max(histogram_NN[location]), c="g")
        axarr[2].legend(["GT", "displ", "h_FM", "h_NN"])
        f.tight_layout()
        plt.savefig("./comparison/" + file_list[location][1][-5:] + ".png")
        plt.close()
    return (line, line_integral)


def compute(displacement, gt):
    displacement = displacement[~np.isnan(gt)]
    gt = gt[~np.isnan(gt)]
    disp = displacement - gt
    print(gt)
    line = compute_AC_curve(filter_to_max(disp, 500))
    line_integral = get_integral_from_line(line[0], line[1])
    streak = get_streak(disp)
    # streak_integral = get_integral_from_line(streak)
    return disp, line, line_integral, streak


def compute_AC_curve(error):
    disp = np.sort(abs(error))
    length = len(error)
    print(length)
    return [disp, np.array(range(length)) / length]


## TODO this is probably incorect since it just summs all the errors therefore not normalised
def get_integral_from_line(values, places=0):
    total = 0
    gaps = np.diff(places)
    for idx, i in enumerate(values):
        total = total + i * places[idx]
    return total


def grade_type(eval_out, gt_file_in, dest):
    print("loading")
    with open(eval_out, 'rb') as handle:
        things_out = pickle.load(handle)
    print("loaded eval data")
    with open(gt_file_in, 'rb') as handle:
        gt_out = pickle.load(handle)
    print("loaded GT")
    compute_to_file(things_out, gt_out, dest)


if __name__ == "__main__":
    print("loading")
    with open(eval_out, 'rb') as handle:
        things_out = pickle.load(handle)
    print("loaded eval data")
    with open(gt_file_in, 'rb') as handle:
        gt_out = pickle.load(handle)
    print("loaded GT")
    compute_with_plot(things_out, gt_out)
