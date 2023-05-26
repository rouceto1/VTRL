import torch as t
from torch.utils.data import Dataset
from torchvision.io import read_image
import random
import kornia as K
import numpy as np
import torchvision
import cv2
from .utils import plot_heatmap
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class StrandsImgPairDataset(Dataset):

    def __init__(self, training_input=None, crop_width=56, training=False):
        super(StrandsImgPairDataset, self).__init__()
        self.width = 512
        self.height = 404
        self.crop_width = crop_width
        self.train = training
        self.training_input = training_input
        ## training_input = file, file, displacement, feature count
        if not self.train:
            self.data = []
            for i, pair in enumerate(self.training_input):
                # if i == 44:
                # print(self.GT[i])
                path1 = pair[0]
                path2 = pair[1]
                self.data.append((path1, path2))
        else:

            temp = self.training_input[:, 2].astype(np.float32) * self.width
            self.disp = temp.astype(int)
            self.fcount1 = self.training_input[:, 3].astype(np.float32).astype(np.int32)
            self.fcount2 = self.training_input[:, 4].astype(np.float32).astype(np.int32)
            self.max_fcount = max(max(self.fcount1), max(self.fcount2))
            self.fcount_threshold = np.percentile([self.fcount1, self.fcount2], 50)
            ##print (GT[0])
            qualifieds = np.array(self.fcount1) >= self.fcount_threshold
            qualifieds2 = np.array(self.fcount2) >= self.fcount_threshold
            qualifieds3 = abs(self.disp) < int(self.width - self.crop_width)
            self.nonzeros = np.count_nonzero(np.logical_and(qualifieds, qualifieds2, qualifieds3))
            print("[+] {} images were qualified out of {} images with {} images not being aligned at all".format(
                self.nonzeros, len(qualifieds), 0.1))
            if self.nonzeros == 0:
                print("[-] no valid selection to teach on. Exiting")
                exit(0)

            self.data = []
            for i, pair in enumerate(self.training_input):
                # if i == 44:
                # print(self.GT[i])
                path1 = pair[0]
                path2 = pair[1]
                if qualifieds[i] and qualifieds2[i]:
                    self.data.append((path1, path2, self.disp[i], i))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        source_img = read_image(self.data[idx][0], mode=torchvision.io.image.ImageReadMode.RGB) / 255.0
        target_img = read_image(self.data[idx][1], mode=torchvision.io.image.ImageReadMode.RGB) / 255.0
        if self.train:
            displ = self.data[idx][2]
            return source_img, target_img, displ, self.data[idx][3]
        else:
            return source_img, target_img


class Strands(StrandsImgPairDataset):
    # crop 64
    # fraction 8
    # smoothnes 3
    # datapath path to the main folder???
    # dsp path to the file with displacements
    # seasosns array of names of the folders with iamfes.
    def __init__(self, crop_width, fraction, smoothness, training_input, training=False):
        super().__init__(training_input=training_input, crop_width=crop_width, training=training)

        self.fraction = fraction
        self.smoothness = smoothness
        # self.flip = K.Hflip()
        self.flip = K.geometry.transform.Hflip()

    def __getitem__(self, idx):
        if self.train:
            source, target, displ, data_idx = super().__getitem__(idx)
            # source[:, :32, -64:] = (t.randn((3, 32, 64)) / 4 + 0.5).clip(0.2, 0.8) # for vlurring out the water mark
            # displ = displ*(512.0/self.width)
            cropped_target, crop_start, original_image = self.crop_img(target, displac=displ)
            if self.smoothness == 0:
                heatmap = self.get_heatmap(crop_start)
            else:
                # TODO tady muze bejt fuckup
                heatmap = self.get_smooth_heatmap(crop_start)
            #plot_heatmap(source, target, cropped_target, displ, heatmap)
            return source, cropped_target, heatmap, data_idx, original_image, displ
        else:
            # croping target when evalution is up to 504 pixels
            source, target = super().__getitem__(idx)
            left = (self.width - self.crop_width) / 2
            right = (self.width - self.crop_width) / 2 + self.crop_width
            return source, target[:, :, int(left):int(right)], target

    def crop_img(self, img, displac):
        # crop - avoid asking for unavailable crop
        if displac >= 0:
            crops = [random.randint(0, int(self.width - self.crop_width - 1) - displac)]
        else:
            crops = [random.randint(0 - displac, int(self.width - self.crop_width - 1))]

        crop_start = random.choice(crops)
        crop_out = crop_start + displac
        # crop_start = random.randint(0, self.width - self.crop_width - 1)
        return img[:, :, crop_start:crop_start + self.crop_width], crop_out, img

    def crop_img_old(self, img, displac):
        # lower and upper bound simoblise the MIDDLE of the possible crops
        if displac == 0:
            lower_bound = self.crop_width / 2
            upper_bound = self.width - self.crop_width / 2
        elif displac > 0:
            lower_bound = self.crop_width / 2
            upper_bound = self.width - self.crop_width / 2 - displac
        elif displac < 0:
            lower_bound = self.crop_width / 2 + abs(displac)
            upper_bound = self.width - self.crop_width / 2
        # print("u  ", upper_bound, lower_bound, dspl, self.crop_width)
        crop_center = random.randint(lower_bound, upper_bound)
        crop_start = int(crop_center - self.crop_width / 2)
        return img[:, :, crop_start:crop_start + self.crop_width], crop_start, img

    def plot_crop_bound(self, im, x1, x2, x3, x4):
        image = im.numpy().transpose(1, 2, 0)
        cv2.line(image, (int(x1), 0), (int(x1), 300), (255, 0, 0), 1)
        cv2.line(image, (int(x2), 0), (int(x2), 300), (255, 255, 0), 1)
        cv2.line(image, (int(x3), 0), (int(x3), 300), (255, 255, 255), 1)
        # cv2.line(image, (int(x4), 0), (int(x4), 300), (255, 255, 255), 1)
        cv2.imshow('image', image)
        cv2.waitKey(0)

    def get_heatmap(self, crop_start):
        frac = self.width // self.fraction
        heatmap = t.zeros(frac).long()
        idx = int((crop_start + self.crop_width // 2) * (frac / self.width))
        heatmap[idx] = 1
        heatmap[idx + 1] = 1
        return heatmap

    def get_smooth_heatmap(self, crop_start):
        surround = self.smoothness * 2
        frac = self.width // self.fraction
        heatmap = t.zeros(frac + surround)
        idx = int((crop_start + self.crop_width // 2) * (frac / self.width)) + self.smoothness
        heatmap[idx] = 1
        idxs = np.array([-1, +1])
        for i in range(1, self.smoothness + 1):
            indexes = list(idx + i * idxs)
            for j in indexes:
                if 0 <= j < heatmap.size(0):
                    heatmap[j] = 1 - i * (1 / (self.smoothness + 1))
        return heatmap[surround // 2:-surround // 2]


if __name__ == '__main__':
    data = StrandsImgPairDataset()
    print(len(data))
    # plot_img_pair(*data[100])
