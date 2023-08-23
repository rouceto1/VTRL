#!/usr/bin/env python3
import marshal as pickle
import cv2
from matplotlib import pyplot as plt
from .helper_functions import *


class Grading:
    def __init__(self):
        self.errors_nn = None
        self.errors_fm = None
        self.AC_fm = None
        self.AC_nn = None
        self.AC_fm_integral = None
        self.AC_nn_integral = None
        self.streak = None
        self.streak_integral = None

    def read_GT(gt):
        gt_disp = []
        # gt_features = []
        gt_place_a = []
        # gt_timeA = []
        gt_place_b = []
        # gt_timeB = []
        # gt_histogram_FM = []
        # gt_histogram_NN = []
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

    def compute(self, displacement, gt, positions=None, plot=True, fig_place=None, hist=None, name=""):
        # displacement_filtered, gt_filtered, count = filter_from_two_arrays_using_thrashold_to_first(displacement,
        #            np.array(gt), 0.5)
        displacement_filtered = filter_to_max(displacement, 1)
        disp = displacement - gt
        max_reached = np.count_nonzero(abs(displacement) > 1)
        max_total = max_reached / len(displacement)
        disp[abs(displacement) > 1] = max_total
        # disp = np.append(disp, 0.5)
        self.AC_fm = compute_AC_curve(disp)

        self.AC_fm_integral = get_integral_from_line(self.AC_fm)
        disp_nn = np.argmax(hist, axis=1) * (1 / 63) - 0.5
        self.AC_nn = compute_AC_curve(disp_nn)
        self.AC_nn_integral = get_integral_from_line(self.AC_nn)
        self.streak = self.get_streak(filter_to_max(disp, 1))
        if plot is True:
            self.plot_all(disp, displacement_filtered, gt, self.AC_fm, self.AC_nn, self.streak, positions, fig_place,
                          name=name)
        self.errors_fm = disp
        self.errors_nn = disp_nn
        self.streak_integral = get_integral_from_line(self.streak)

    ##TODO do this function
    def get_streak(self, disp):
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

    def plot_all(self, disp, displacement_filtered, gt_filtered, line, line_2, streak, positions, save, name=""):
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

        # plot3.ylabel("Place visited", fontsize=18)
        # plot3.xlabel("Timestamp [s]", fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(save, "input.png"), dpi=800)
        pos = plot3.get_position()
        x = pos.x0 + 0.4
        y = pos.y0 + 0.4
        plt.figtext(x, y, save.split("/")[-1])
        # plt.show()
        # plt.close()
        #


def compute_AC_to_uncertainty(estimates, gt, hist_nn, hist_fm, matches):
    #TODO incorporate this to calcualations
    estimates_filtered = filter_to_max(estimates, 1)
    z = np.array([])
    x = np.array([])
    y = np.array([])
    disp = estimates_filtered - gt
    # disp = np.append(disp, 1)
    # matches = np.append(matches, 1600)
    for i in range(1600):
        z = np.append(z, (matches < i).sum())
        l = np.array(compute_AC_curve(disp[(matches < i)]))
        x = np.append(x, l[0])
        y = np.append(y, l[1])

    z = np.array(z)
    # x = np.array(x)
    # y = np.array(y)

    line = compute_AC_curve(disp)
    ig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, z, y, c=z, cmap='Greens')
    plt.show()
    pass


def filter_from_two_arrays_using_thrashold_to_first(a1, a2, threshold):
    count = sum(x > threshold for x in abs(a1))
    # disp[abs(disp) > threshold] = 0
    a1_o = a1[(abs(a1) <= threshold)]
    a2_o = a2[(abs(a1) <= threshold)]
    return a1_o, a2_o, count


def compute_AC_curve(error):
    # error = [-2.548,458,5,0,5684,-5684.23]
    length = len(error)
    if length == 0:
        return [[0], [0]]
    disp = np.sort(abs(error))
    percentages = np.array(range(length)) / length
    cut = (disp <= 0.5).sum()
    return [np.append(0, np.append(disp[:cut], 0.5)), np.append(0, np.append(percentages[:cut], 1))]


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


def grade_type(mission, _GT=None, estimates=None):
    if estimates is None:
        print("from " + str(mission.c_strategy.grading_path))
        with open(mission.c_strategy.grading_path, 'rb') as handle:
            estimates = pickle.load(handle)
    GT_versions = ["strands", "grief"]
    slices = [[3054, None], [None, 3053], [None, None]]
    for index, G in enumerate(GT_versions):
        file_list, displacements, feature_count_l, feature_count_r, matches, histograms, hist_nn = estimates[0]
        file_list = file_list[slices[index][0]:slices[index][1]]
        displacements = displacements[slices[index][0]:slices[index][1]]
        matches = matches[slices[index][0]:slices[index][1]]
        histograms = histograms[slices[index][0]:slices[index][1]]
        hist_nn = hist_nn[slices[index][0]:slices[index][1]]
        GT_reduced = _GT[slices[index][0]:slices[index][1]]
        gt = read_gt_file(file_list, GT_reduced)
        mission.c_strategy.grading = Grading()
        mission.c_strategy.grading.compute(displacements, gt, mission.mission_folder, mission.c_strategy.plan,
                                           fig_place=mission.plot_folder, hist_nn=hist_nn, hist_fm=histograms,
                                           matches=matches, name=G)


def filter_to_max(lst, threshold):
    lst[lst > threshold] = threshold
    lst[lst < -threshold] = -threshold
    return lst

