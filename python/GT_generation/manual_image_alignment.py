import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def equalize_color(img):
    R, G, B = cv2.split(img)

    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)

    return cv2.merge((output1_R, output1_G, output1_B))


class Annotate:
    def __init__(self, path_GT, path_GT_out, path_dataset):
        self.path = path_GT
        self.path_out = path_GT_out
        with open(self.path, 'rb') as handle:
            self.list_all = pickle.load(handle)
        self.dataset_path = path_dataset

    def show_GT(self):
        for entry in tqdm(self.list_all):
            if not "cest" in entry[4]:
                pass
                #continue
            if  entry[-1] <1:
                pass
                continue
            image1 = cv2.imread(os.path.join(self.dataset_path, entry[4], entry[5]) + ".png")
            # image1 = cv2.cvtColor(image1, cv2.COLOR)
            image2 = cv2.imread(os.path.join(self.dataset_path, entry[6], entry[7]) + ".png")
            # image2 = cv2.cvtColor(image2, cv2.COLOR)
            shift = entry[0]
            image_draw, image11, image22 = self.draw_overlay(image1, image2, shift, equalize=True,
                                                             overaly=0)
            self.draw(image_draw, image11, image22, names=["+", "+"])

    def annotate(self):
        # program takes a list_all of two images and a percantage shift between them. Shows both images overlayed both images with the shift applied.
        # Using arowkeys shift can be fine tuned and then using space key saved as a new value
        # using number row to select a specific overlay settings
        # the program is intended to be used to annotate the dataset with the correct shift values
        count = 0
        save = False
        overlay = 0
        equalize = False
        last_entry = None
        for number, entry in enumerate(self.list_all[0][0]):
            if len(entry) > 10:  # TODO proper size of each line
                # continue
                pass
            if not "testi" in entry[0]:
                pass
                # continue

            # entry[4] = entry[4][:-1] + "0" + entry[4][-1:]
            # entry[6] = entry[6][:-1] + "0" + entry[6][-1:]
            do_GT = False
            if do_GT:
                image1 = cv2.imread(os.path.join(self.dataset_path, entry[4], entry[5]) + ".bmp")
                # image1 = cv2.cvtColor(image1, cv2.COLOR)
                image2 = cv2.imread(os.path.join(self.dataset_path, entry[6], entry[7]) + ".bmp")
                # image2 = cv2.cvtColor(image2, cv2.COLOR)
                shift = entry[0]
            else:
                image1 = cv2.imread("/home" + entry[0][11:] + ".png")

                image2 = cv2.imread("/home" + entry[1][11:] + ".png")
                if image1 is None or image2 is None:
                    continue
                shift = self.list_all[0][1][number]
            print(shift)
            image11 = image1

            while True:

                image_draw, image11, image22 = self.draw_overlay(image1, image2, shift, equalize=equalize,
                                                                 overaly=overlay)
                k = self.draw(image_draw, image11, image22)
                if k == 114:  # R as remove last entry
                    last_entry = last_entry[:-1]
                if k == 27:  # esc3
                    save = True
                    break
                if k == 99:  # c as continue
                    break
                if k == 81:  # left
                    shift = shift - 1 / image1.shape[1]
                    shift = shift % 1
                if k == 83:  # right
                    shift = shift + 1 / image1.shape[1]
                    shift = shift % 1
                if k == 85:  # page up (lots left)
                    shift = shift - 20 / image1.shape[1]
                    shift = shift % 1
                if k == 86:  # page down (lots right)
                    shift = shift + 20 / image1.shape[1]
                    shift = shift % 1
                if k == 13:  # enter
                    entry[0] = shift
                    entry.append(1)
                    last_entry = entry
                    break
                if 60 > k >= 48:  # 1
                    overlay = k - 48
                if k == 101:  # pressing E
                    equalize = not equalize
                if k == 115:  # S as shit
                    entry[0] = shift
                    entry.append(-1)
                    last_entry = entry
                    break
                if k == 102:  # F as totaly Fucked
                    entry[0] = shift
                    entry.append(-2)
                    last_entry = entry
                    break
                if k == 97:  # A ambiguous
                    entry[0] = shift
                    entry.append(0)
                    last_entry = entry
                    break
            last_entry = entry
            if save:
                break
        # self.save_GT(self.list_all)

    def save_GT(self, data):
        print("saving")
        with open(self.path_out, 'wb') as handle:
            pickle.dump(data, handle)

    def shift_image_by_pixels(self, image, percent):
        # translate image by set amount of pixels and roll over
        # eprint(percent)
        pixels = percent * image.shape[1]

        return np.roll(image, int(pixels), axis=1)

    def draw_overlay(self, image1, image2, shift, equalize=False, overaly=0):
        # combines two images to one with same dimensions with the shift applied
        # overlay is a number between 0 and 9 that selects the overlay mode
        # 0 = both images normal
        # 1 = both images red channel only
        # 2 = both images green channel only
        # 3 = both images blue channel only
        # 4 = both images only edges
        # 5 = both images histogram equalized
        # 6 = image1 histogram equalized
        # 7 = image2 histogram equalized
        # 8 = image1 histogram equalized and image2 red channel only
        # 9 = image1 histogram equalized and image2 green channel only

        if equalize:
            image1 = equalize_color(image1)
            image2 = equalize_color(image2)
        image2 = self.shift_image_by_pixels(image2, shift)
        if overaly == 0:
            out0 = image1 - image2
        elif overaly == 1:
            out0 = image1[:, :, 0] - cv2.Canny(image2, 100, 200)
        elif overaly == 2:
            out0 = image1[:, :, 1] - image2[:, :, 1]
        elif overaly == 3:
            out0 = image1[:, :, 2] - image2[:, :, 2]
        elif overaly == 4:
            out0 = cv2.Canny(image1, 100, 200) - cv2.Canny(image2, 100, 200)
        elif overaly == 6:
            out0 = cv2.Canny(image1, 100, 200) + cv2.Canny(image2, 100, 200)
        elif overaly == 7:
            out0 = cv2.Canny(image2, 100, 200) - cv2.Canny(image1, 100, 200)
        elif overaly == 8:
            out0 = equalize_color(image1) - image2
        elif overaly == 9:
            out0 = image1 - equalize_color(image2)
        else:
            out0 = image1 - image2
        out1 = image1
        out2 = image2
        return out0, out1, out2

    def draw(self, image_modified, image1, image2, names=["i1", "i2"]):
        # Function to draw image in a window using cv2 and waits for a keypress and returns the key pressed
        image = np.concatenate((image1, image2), axis=0)
        cv2.imshow(names[0], image_modified)
        cv2.imshow(names[1], image)
        k = cv2.waitKey(0)
        return k

    def combine_annotations(self):
        # takes annotations of all postiions agianst one specific postion and makes all possible pairs that are of the same place
        # returns a new annotation list containg all viable pairs where nither has been marked as bad
        # format of annotation list is [shift, features1, features2, matches, path11,path12,path21,path22,[histogram1],[histogram2],good/bad]
        # bad files are marked with -1 or less as last element
        # good files are marked with 1
        # ambiguous files are marked with 0
        # returns a list of lists with the same fromat as input
        self.list_all_new = []
        self.list_all_filtered = []
        too_large = 0
        for i in self.list_all:
            if i[-1] >= 1:
                self.list_all_filtered.append(i)
        already_matched = []
        for entry in tqdm((self.list_all_filtered)):
            for entry2 in (self.list_all_filtered):
                if not entry[6][-1] == entry2[6][-1]:
                    continue
                if entry[7] == entry2[7]:
                    continue
                new_entry = []
                combined_shift = entry2[0] - entry[0]
                if abs(combined_shift) >= 1:
                    too_large += 1
                    continue
                if abs(combined_shift) >= 0.5:
                    pass
                    too_large += 1
                    continue
                #if [entry[0], entry2[0]] in already_matched:
                #    continue
                new_entry.append(combined_shift)
                new_entry.append(entry[2])
                new_entry.append(entry2[2])
                combined_matches = min(entry[3], entry2[3])
                new_entry.append(combined_matches)
                new_entry.append(entry[6])
                new_entry.append(entry[7])
                new_entry.append(entry2[6])
                new_entry.append(entry2[7])
                combined_histogram1 = entry[8] + entry2[8]
                new_entry.append(combined_histogram1)
                combined_histogram2 = entry[9] + entry2[9]
                new_entry.append(combined_histogram2)
                new_entry.append(1)
                self.list_all_new.append(new_entry)
                #already_matched.append([entry[0], entry2[0]])
        print("too large: ", too_large)
        self.save_GT(self.list_all_new)


def test(path1, path2):
    with open(path1, 'rb') as handle:
        path1_p = pickle.load(handle)
    with open(path2, 'rb') as handle:
        path2_p = pickle.load(handle)
    for i, p in enumerate(path1_p):
        if path1_p[i] is not path2_p[i]:
            print(path1_p[i])
            print(path2_p[i])


class Compare:
    def __init__(self, path_1, path_2, path_dataset):
        self.path_1 = path_1
        self.path_2 = path_2
        with open(self.path_1, 'rb') as handle:
            self.list_all1 = pickle.load(handle)
        with open(self.path_2, 'rb') as handle:
            self.list_all2 = pickle.load(handle)
        self.dataset_path = path_dataset

    def show_aligned(self):
        for number, entry in enumerate(self.list_all1[0][0]):
            image1 = cv2.imread("/home" + entry[0][11:] + ".png")
            image2 = cv2.imread("/home" + entry[1][11:] + ".png")
            image3 = cv2.imread("/home" + entry[0][11:] + ".png")
            image4 = cv2.imread("/home" + entry[1][11:] + ".png")

            shift1 = self.list_all1[0][1][number]
            shift2 = self.list_all2[0][1][number]
            image1 = equalize_color(image1)
            image2 = equalize_color(image2)
            image3 = equalize_color(image3)
            image4 = equalize_color(image4)
            pixels1 = shift1 * image1.shape[1]
            pixels2 = shift2 * image2.shape[1]
            image2 = np.roll(image2, int(pixels1), axis=1)
            image4 = np.roll(image4, int(pixels2), axis=1)

            self.draw(image1, image2, image3, image4, names=[self.path_1[32:38], self.path_2[32:38]])

    def draw(self, image1, image2, image3, image4, names=["i1", "i2"]):
        imagea = np.concatenate((image1, image2), axis=0)
        imageb = np.concatenate((image3, image4), axis=0)
        cv2.imshow(names[0], imagea)
        cv2.imshow(names[1], imageb)
        k = cv2.waitKey(0)
        return k


def create_GT_grief(root_path, width):
    # format of annotation list is [shift, features1, features2, matches, path11,path12,path21,path22,[histogram1],[histogram2],good/bad]
    new_GT = []
    idx = 0
    with os.walk(root_path) as file:  # TODO fix the file walk
        if not file.contains("place_00") or file.contains("bmp"):
            gt = []
            disp = np.loadtxt(os.path.join(root_path, "input.txt"), int)
            paths = os.path.split(file)
            id = int(paths[0].split(".")[0])
            shift = disp[id] / width
            gt.append(shift)
            gt.append(0, 0, 0)  # features, features and matches
            gt.append("place_00")
            gt.append(id)
            gt.append(paths[-2])
            gt.append(id)
            gt.append([])  # histogram
            gt.append([])  # histogram
            gt.append(1)
            idx = idx + 1
            new_GT.append(gt)
    with open(os.path.join(root_path, "GT.pickle"), 'wb') as handle:
        pickle.dump(new_GT, handle)


if __name__ == "__main__":
    pickle_file_with_image_paths_and_shifts = os.path.join("/home/rouceto1/datasets/grief_jpg/GT_redone.pickle")
    #pickle_file_with_image_paths_and_shifts = os.path.join("/home/rouceto1/VTRL/datasets/grief_jpg/GT_merge.pickle")
    #pickle_file_with_image_paths_and_shifts = os.path.join("/home/rouceto1/VTRL/experiments/0.02_0/estimates.pickle")
    pickle_file_to_save_to = os.path.join("/home/rouceto1/datasets/grief_jpg/GT_redone_best.pickle")
    dataset_path = os.path.join("/home/rouceto1/datasets/grief_jpg")
    # test(pickle_file_with_image_paths_and_shifts, pickle_file_to_save_to)
    # create_GT_grief("/home/rouceto1/datasets/grief_jpg/cestlice_reduced", 1024)
    annotation = Annotate(pickle_file_with_image_paths_and_shifts, pickle_file_to_save_to, dataset_path)
    # annotation.annotate()
    annotation.combine_annotations()
    #annotation.show_GT()

    pickle_file_with_image_paths_and_shifts1 = os.path.join("/home/rouceto1/VTRL/experiments/0.02_0/estimates.pickle")
    pickle_file_with_image_paths_and_shifts2 = os.path.join("/home/rouceto1/VTRL/experiments/0.08_1/estimates.pickle")
    #comp = Compare(pickle_file_with_image_paths_and_shifts1, pickle_file_with_image_paths_and_shifts2, dataset_path)
    #comp.show_aligned()

    #
