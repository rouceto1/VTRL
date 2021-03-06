from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn.metrics import auc, roc_curve
from matplotlib import image
import matplotlib.patches as patches


mpl.use('Qt5Agg')


DATASET_NAME = "carlevaris"
DATASET_PLOT = "carlevaris"
ERROR_CAP = 512


def error_distribution(files: [str], names: [str]):
    for file in files:
        array = np.genfromtxt(file, delimiter=',')
        array = np.array(sorted(abs(np.clip(array, -ERROR_CAP, ERROR_CAP))))
        print(auc(array, np.linspace(0, 1, len(array))))
        out_fracs = np.zeros(ERROR_CAP)
        for i in range(ERROR_CAP):
            out_fracs[i] = np.sum(array < i)/float(len(array))
        print(out_fracs[50])
        plt.plot(out_fracs, linewidth=2)

    plt.title("Accuracies on " + DATASET_NAME + " dataset", fontsize=16)
    plt.grid()
    plt.xlabel("Registration error threshold [px]", fontsize=14)
    plt.ylabel("Prob. of correct registration [-]", fontsize=14)
    # plt.xscale("log")
    plt.xlim([10, 70])
    plt.ylim([0.50, 1])
    plt.legend(names, loc=4, fontsize=16)
    # plt.savefig("./accuracy_" + DATASET_NAME + ".eps")
    plt.show()


def plot_cutout_region(img_path: str, region1: [[int], [int]], region2: [[int], [int]]):
    img = image.imread(img_path)
    f, axarr = plt.subplots(2)
    axarr[0].imshow(img, aspect="auto")
    axarr[0].axvspan(region1[0][0], region1[0][1], color='green', alpha=0.5)
    axarr[0].axvspan(region1[1][0], region1[1][1], color='green', alpha=0.5)
    axarr[0].title.set_text("Cutout restriction for coarse dataset")
    axarr[1].imshow(img, aspect="auto")
    axarr[1].axvspan(region2[0][0], region2[0][1], color='green', alpha=0.5)
    axarr[1].axvspan(region2[1][0], region2[1][1], color='green', alpha=0.5)
    rect = patches.Rectangle((440, 0), 40, 40, linewidth=1, edgecolor='k', facecolor='k')
    axarr[1].add_patch(rect)
    axarr[1].title.set_text("Cutout restriction for rectified dataset")
    f.tight_layout()
    plt.savefig("./cutouts.png")
    plt.close()


def plot_annotation_ambiguity(img_path1: str, img_path2: str):
    img1 = image.imread(img_path1)
    img2 = image.imread(img_path2)
    f, axarr = plt.subplots(2)
    axarr[0].imshow(img1, aspect="auto")
    axarr[1].imshow(img2, aspect="auto")
    axarr[0].axvline(x=390, ymin=0, ymax=420, c="b")
    axarr[1].axvline(x=400, ymin=0, ymax=420, c="b")
    axarr[0].axvline(x=750, ymin=0, ymax=420, c="r")
    axarr[1].axvline(x=715, ymin=0, ymax=420, c="r")

    f.suptitle("Annotation ambiguity")
    f.tight_layout()
    plt.savefig("./ambiguity.png")
    plt.close()


if __name__ == '__main__':
    # plot_annotation_ambiguity("/home/zdeeno/Documents/Datasets/grief_jpg/carlevaris/season_00/000000475.jpg",
    #                           "/home/zdeeno/Documents/Datasets/grief_jpg/carlevaris/season_01/000000475.jpg")
    # plot_cutout_region("/home/zdeeno/Documents/Datasets/nordland_rectified/spring/008409.png", [[80, 180], [511 - 180, 511 - 80]], [[0, 180], [511 - 180, 511]])
    error_distribution([# "/home/zdeeno/Documents/Work/GRIEF/results/grief_errors.csv",
                        "/home/zdeeno/Documents/Work/GRIEF/results/grief_errors_" + DATASET_PLOT.lower() + ".csv",
                        "/home/zdeeno/Documents/Work/GRIEF/results/sift_errors_" + DATASET_PLOT.lower() + ".csv",
                        # "/home/zdeeno/Documents/Work/alignment/results_siam/eval_model_70/" + DATASET_PLOT.lower() + "_errors.csv",
                        "/home/zdeeno/Documents/Work/alignment/results_siam_cnn/eval_model_47/" + DATASET_PLOT.lower() + "_errors.csv",
                        "/home/zdeeno/Documents/Work/SuperPointPretrainedNetwork/superpixel_errors_" + DATASET_PLOT.lower()],
                         ["grief", "sift", "siamese_old", "superpoint"])

    # error_distribution([# "/home/zdeeno/Documents/Work/GRIEF/results/grief_errors.csv",
    #                     "/home/zdeeno/Documents/Work/alignment/results_siam/eval_model_20/" + DATASET_PLOT.lower() + "_errors.csv",
    #                     "/home/zdeeno/Documents/Work/alignment/results_siam/eval_model_50/" + DATASET_PLOT.lower() + "_errors.csv",
    #                     "/home/zdeeno/Documents/Work/alignment/results_siam/eval_model_60/" + DATASET_PLOT.lower() + "_errors.csv",
    #                     "/home/zdeeno/Documents/Work/alignment/results_siam/eval_model_70/" + DATASET_PLOT.lower() + "_errors.csv"],
    #                     # "/home/zdeeno/Documents/Work/SuperPointPretrainedNetwork/superpixel_errors_" + DATASET_PLOT.lower()],
    #                      ["20", "50", "60", "70"])
