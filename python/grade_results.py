#!/usr/bin/env python3
import marshal as pickle

import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
from .helper_functions import *
from csv import writer
from pathlib import Path
import time
import cv2
from mpl_toolkits import mplot3d


def filter_to_max(lst, threshold):
    lst[lst > threshold] = threshold
    lst[lst < -threshold] = -threshold
    return lst


def add_img_to_plot(plot, img_path):
    img = image.imread(img_path + ".bmp")
    plot.imshow(img, aspect="auto")


def read_GT(gt):
    gt_disp = []
    gt_features = []
    gt_place_a = []
    gt_timeA = []
    gt_place_b = []
    gt_timeB = []
    gt_histogram_FM = []
    gt_histogram_NN = []
    for place in gt:
        gt_disp.append(place[0])
        # gt_features.append(place[1])
        gt_place_a.append(place[2] + "/" + place[3])
        # gt_timeA.append(place[3])
        gt_place_b.append(place[4] + "/" + place[5])
        # gt_timeB.append(place[5])
        # gt_histogram_FM.append(place[6])
        # gt_histogram_NN.append(place[7])
    return gt_disp, gt_place_a, gt_place_b


##TODO do this function
def get_streak(disp):
    streak = []
    poses = []
    for i in range(0, 100, 1):
        length = 0
        temp = 0
        poses.append(i / 100)
        for dis in abs(disp):
            if dis < i / 100.0:
                temp += 1 / len(disp)
            else:
                if temp > length:
                    length = temp
                temp = 0
        if temp > length:
            length = temp
        streak.append(length)
    return [poses, streak]


def compute_AC_to_uncertainty(estimates, gt, hist_nn, hist_fm, matches):

    estimates_filtered, gt_filtered, count = filter_from_two_arrays_using_thrashold_to_first(estimates,
                                                                                             np.array(gt), 0.5)
    gt_filtered, estimates_filtered, count2 = filter_from_two_arrays_using_thrashold_to_first(gt_filtered,
                                                                                              estimates,
                                                                                              0.5)
    assert (count == count2 == 0)
    z = np.array([])
    x = np.array([])
    y = np.array([])
    disp = estimates_filtered - gt_filtered
    #disp = np.append(disp, 1)
    #matches = np.append(matches, 1600)
    for i in range(1600):
        z = np.append(z,(matches < i).sum())
        l = np.array(compute_AC_curve(disp[(matches < i)]))
        x = np.append(x,l[0])
        y = np.append(y,l[1])

    z = np.array(z)
    #x = np.array(x)
    #y = np.array(y)

    line = compute_AC_curve(disp)
    ig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, z, y, c=z, cmap='Greens')
    plt.show()
    pass


def compute_to_file(estimates, gt, dist, positions, plot=True, fig_place=None, hist_nn=None, hist_fm=None, matches=None,
                    name=""):
    line_out = os.path.join(dist, "line_" + name + ".pkl")
    streak_out = os.path.join(dist, "streak_" + name + ".pkl")
    # file_list, histogram_fm, histogram_nn, feature_count, displacement, gt_disp = load_data(data, gt)
    #compute_AC_to_uncertainty(estimates, gt, hist_nn, hist_fm, matches)
    errors, line, line_integral, line_2, line_2_integral, streak, streak_integral = compute(estimates, gt,
                                                                                            positions=positions,
                                                                                            plot=plot,
                                                                                            fig_place=fig_place,
                                                                                            hist=hist_nn, name=name)

    with open(line_out, 'wb') as hand:
        pickle.dump(line_2, hand)
        print("Line written " + str(line_out))
    with open(streak_out, 'wb') as hand:
        pickle.dump(streak, hand)
        print("streak " + str(streak_out))
    return round(sum(errors) / len(errors), 5), round(streak_integral, 5), round(line_integral, 5), round(
        line_2_integral, 5)


def compute_with_plot(data, gt):  # TODO full redo of this...
    file_list, histogram_FM, histogram_NN, feature_count, displacement, gt_disp = load_data(data, gt)
    disp, line, line_integral, line_2, line_2_integral, streak, streak_integral = compute(displacement, gt_disp)
    for location in range(50, len(file_list)):
        f, axarr = plt.subplots(3)
        for i in [0, 1]:
            add_img_to_plot(axarr[i], file_list[location][i])
        # print(file_list[location])
        r1 = range(-630, 630, 1260 // 63)  # for FM since it is using full imagees
        r2 = range(-504, 504, 1008 // 63)  # for NN since it is using reduced images
        axarr[2].axvline(x=gt_disp[location], ymin=0, ymax=1, c="b", ls="--")
        axarr[2].axvline(x=displacement[location], ymin=0, ymax=1, c="k", ls="--")
        axarr[2].plot(r1, histogram_FM[location] / max(histogram_FM[location]), c="r")
        axarr[2].plot(r1, histogram_NN[location] / max(histogram_NN[location]), c="g")
        axarr[2].legend(["GT", "displ", "h_FM", "h_NN"])
        f.tight_layout()
        plt.savefig("./comparison/" + file_list[location][1][-5:] + ".png")
        plt.close()
    return line, line_integral


def filter_from_two_arrays_using_thrashold_to_first(a1, a2, threshold):
    count = sum(x > threshold for x in abs(a1))
    # disp[abs(disp) > threshold] = 0
    a1_o = a1[(abs(a1) <= threshold)]
    a2_o = a2[(abs(a1) <= threshold)]
    return a1_o, a2_o, count


def plot_all(disp, displacement_filtered, gt_filtered, line, line_2, streak, positions, save, name=""):
    plot1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
    plot2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
    plot3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    plot1.plot(disp, label='diff', linewidth=0.2)
    plot1.plot(displacement_filtered, label='displacement', linewidth=0.2)
    plot1.plot(gt_filtered, label='gt', linewidth=0.2)
    plot1.legend()
    plot1.set_title("raw displacements")

    plot2.plot(line[0], line[1], label='AC')
    plot2.plot(line_2[0], line_2[1], label='AC_filtered')
    plot2.plot(streak[0], streak[1], label="streak")
    plot2.set_title("metrics")
    plot2.legend()

    interpolated = []
    for pose in positions:
        interpolated.append(pose)
    locs, labels = plt.xticks()
    plot3.plot(interpolated)
    # plot3.ylabel("Place visited", fontsize=18)
    # plot3.xlabel("Timestamp [s]", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(save, "plots", "input_" + name + ".png"), dpi=800)
    pos = plot3.get_position()
    x = pos.x0 + 0.4
    y = pos.y0 + 0.4
    plt.figtext(x, y, save.split("/")[-1])
    # plt.show()
    # plt.close()
    #


def compute(displacement, gt, positions=None, plot=True, fig_place=None, hist=None, name=""):
    #displacement_filtered, gt_filtered, count = filter_from_two_arrays_using_thrashold_to_first(displacement,
                                                                                    #            np.array(gt), 0.5)
    displacement_filtered = filter_to_max(displacement, 1)
    #assert (count == count2 == 0)

    disp = displacement_filtered - gt
    disp = np.append(disp, 1)
    line = compute_AC_curve(displacement_filtered - gt)
    line_2 = compute_AC_curve(disp)
    line_integral = get_integral_from_line(line)
    line_2_integral = get_integral_from_line(line_2)
    streak = get_streak(filter_to_max(displacement - gt, 1))
    if plot is True:
        plot_all(disp, displacement_filtered, gt, line, line_2, streak, positions, fig_place, name=name)

    streak_integral = get_integral_from_line(streak)
    return disp, line, line_integral, line_2, line_2_integral, streak, streak_integral


def compute_AC_curve(error):
    # error = [-2.548,458,5,0,5684,-5684.23]
    length = len(error)
    if length == 0:
        return [[0], [0]]
    disp = np.sort(abs(error))
    return [disp, np.array(range(length)) / length]


def get_integral_from_line(values):
    # Fucntion returns integral of numerical array
    # values=[[1,3,5,3,4,81,2,6,88,52,5,2,5,-5],[1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
    integral = np.trapz(values[1], values[0])  ##DONE AND PROBABLY WORKS
    return integral


def show_estiamtes(file_list, displacements, feature_count_l, feature_count_r, matches, histograms, hist_nn, gt):
    # use opencv to compare image pairs form file_list shifted by displacement and shifted by gt value

    for i in range(len(file_list)):
        # if not "testing" in file_list[i][0] :
        #    continue
        img1 = cv2.imread("/home" + file_list[i][0][11:] + ".png")
        img2 = cv2.imread("/home" + file_list[i][1][11:] + ".png")
        imgm2 = np.roll(img2, int(gt[i] * img2.shape[1]), axis=1)
        image1 = np.concatenate((img1, imgm2), axis=0)
        cv2.imshow("plus", image1)
        img2 = np.roll(img2, int(-gt[i] * img2.shape[1]), axis=1)
        image2 = np.concatenate((img1, img2), axis=0)
        cv2.imshow("minus", image2)
        cv2.waitKey(0)


def grade_type(dest, positions=None, estimates_file=None, _GT=None, estimates=None, time_elapsed=None, data_count=None,
               GT_name=""):
    print("recieve offset estiamtes")
    if estimates is None:
        print("from " + str(estimates_file))
        with open(estimates_file, 'rb') as handle:
            estimates = pickle.load(handle)
    experiemnt_name = os.path.basename(os.path.normpath(dest))
    path = Path(dest).parent

    GT_versions = ["strands", "grief", "all"]
    slices = [[3054, None], [None, 3053], [None, None]]
    exp_time = int(time.time() - time_elapsed)
    for index, G in enumerate(GT_versions):
        file_list, displacements, feature_count_l, feature_count_r, matches, histograms, hist_nn = estimates[0]
        file_list = file_list[slices[index][0]:slices[index][1]]
        displacements = displacements[slices[index][0]:slices[index][1]]
        matches = matches[slices[index][0]:slices[index][1]]
        histograms = histograms[slices[index][0]:slices[index][1]]
        hist_nn = hist_nn[slices[index][0]:slices[index][1]]
        GT_reduced = _GT[slices[index][0]:slices[index][1]]
        gt = read_gt_file(file_list, GT_reduced)

        out = [experiemnt_name, exp_time,
               *compute_to_file(displacements, gt, dest, positions, fig_place=dest, hist_nn=hist_nn, hist_fm=histograms,
                                matches=matches, name=G),
               data_count[0], data_count[1], G]

        with open(path / 'output.csv', 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(out)
            f_object.close()


if __name__ == "__main__":
    annotation_file = "1-2-fast-grief.pt"
    GT_file = annotation_file + "_GT_.pickle"
    evaluation_prefix = "/home/rouceto1/datasets/strands_crop"
    gt_file_in = os.path.join(evaluation_prefix, GT_file)
    dataset_path = "/home/rouceto1/datasets/strands_crop/training_Nov"
    weights_file = "exploration.pt"
    eval_out_file = weights_file + "_eval.pickle"
    eval_out = os.path.join(dataset_path, eval_out_file)
    print("loading")
    with open(eval_out, 'rb') as handle:
        things_out = pickle.load(handle)
    print("loaded eval data")
    with open(gt_file_in, 'rb') as handle:
        gt_out = pickle.load(handle)
    print("loaded GT")
    compute_with_plot(things_out, gt_out)
