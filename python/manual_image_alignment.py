import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


class Annotate:
    def __init__(self, path_GT, path_GT_out, path_dataset):
        self.path = path_GT
        self.path_out = path_GT_out
        with open(self.path, 'rb') as handle:
            self.list_all = pickle.load(handle)
        self.dataset_path = path_dataset

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
        for entry in tqdm(self.list_all):
            if len(entry) > 10:  # TODO proper size of each line
                #continue
                pass
            image1 = cv2.imread(os.path.join(self.dataset_path, entry[4], entry[5]) + ".bmp")
            # image1 = cv2.cvtColor(image1, cv2.COLOR)
            image2 = cv2.imread(os.path.join(self.dataset_path, entry[6], entry[7]) + ".bmp")
            # image2 = cv2.cvtColor(image2, cv2.COLOR)
            shift = entry[0]
            print(shift)
            image11 = image1

            while True:

                image_draw, image11, image22 = self.draw_overlay(image1, image2, shift, equalize=equalize,
                                                                 overaly=overlay)
                k = self.draw(image_draw, image11, image22)
                if k == 114: # R as remove last entry
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
        self.save_GT(self.list_all)

    def save_GT(self, data):
        print("saving")
        with open(self.path_out, 'wb') as handle:
           pickle.dump(data, handle)
        



    def shift_image_by_pixels(self, image, percent):
        # translate image by set amount of pixels and roll over
        #eprint(percent)
        pixels = percent * image.shape[1]

        return np.roll(image, int(pixels), axis=1)

    def equalize_color(self, img):
        R, G, B = cv2.split(img)

        output1_R = cv2.equalizeHist(R)
        output1_G = cv2.equalizeHist(G)
        output1_B = cv2.equalizeHist(B)

        return cv2.merge((output1_R, output1_G, output1_B))

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
            image1 = self.equalize_color(image1)
            image2 = self.equalize_color(image2)
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
            out0 = self.equalize_color(image1) - image2
        elif overaly == 9:
            out0 = image1 - self.equalize_color(image2)
        else:
            out0 = image1 - image2
        out1 = image1
        out2 = image2
        return out0, out1, out2

    def draw(self, image_modified, image1, image2):
        # Function to draw image in a window using cv2 and waits for a keypress and returns the key pressed
        image = np.concatenate((image1, image2), axis=0)
        cv2.imshow('image1', image_modified)
        cv2.imshow('image2', image)
        k = cv2.waitKey(0)
        return k

    def combine_annotations(self):
        #takes annotations of all postiions agianst one specific postion and makes all possible pairs that are of the same place
        # returns a new annotation list containg all viable pairs where nither has been marked as bad
        #format of annotation list is [shift, features1, features2, matches, path11,path12,path21,path22,[histogram1],[histogram2],good/bad]
        #bad files are marked with -1 or less as last element
        #good files are marked with 1
        #ambiguous files are marked with 0
        #returns a list of lists with the same fromat as input
        self.list_all_new = []
        self.list_all_filtered = []
        for i in self.list_all:
            if i[-1] >= -3:
                self.list_all_filtered.append(i)
        already_matched = []
        for entry in tqdm((self.list_all_filtered)):
            for entry2 in (self.list_all_filtered):
                if not entry[6][-1] == entry2[6][-1]:
                    continue
                if entry2[6] + entry2[7] in already_matched:
                    continue
                if entry[7] == entry2[7]:
                    continue
                new_entry = []
                combined_shift = entry[0] - entry2[0] % 1
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
            already_matched.append(entry[6]+entry[7])
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

def create_GT_grief(root_path, width):
    # format of annotation list is [shift, features1, features2, matches, path11,path12,path21,path22,[histogram1],[histogram2],good/bad]
    new_GT = []
    idx = 0
    with os.walk(root_path) as file: #TODO fix the file walk
        if not file.contains("place_00") or file.contains("bmp"):
            gt = []
            disp = np.loadtxt(os.path.join(root_path, "input.txt"), int)
            paths = os.path.split(file)
            id = int(paths[0].split(".")[0])
            shift = disp[id] / width
            gt.append(shift)
            gt.append(0, 0, 0) # features, features and matches
            gt.append("place_00")
            gt.append(id)
            gt.append(paths[-2])
            gt.append(id)
            gt.append([]) # histogram
            gt.append([]) # histogram
            gt.append(1)
            idx = idx + 1
            new_GT.append(gt)
    with open(os.path.join(root_path, "GT.pickle"), 'wb') as handle:
        pickle.dump(new_GT, handle)


if __name__ == "__main__":
    pickle_file_with_image_paths_and_shifts = os.path.join("/home/rouceto1/datasets/strands_crop/GT_redone.pickle")
    pickle_file_to_save_to = os.path.join("/home/rouceto1/datasets/strands_crop/GT_redone_all.pickle")
    dataset_path = os.path.join("/home/rouceto1/datasets/strands_crop")
    #test(pickle_file_with_image_paths_and_shifts, pickle_file_to_save_to)
    create_GT_grief("/home/rouceto1/datasets/grief_jpg/cestlice_reduced", 1024)
    #annotation = Annotate(pickle_file_with_image_paths_and_shifts, pickle_file_to_save_to, dataset_path)
    #annotation.combine_annotations()
    #annotation.annotate()