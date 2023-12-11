import functools

import torch as t
from torch.utils.data import Dataset
from torchvision.io import read_image
import random
import kornia as K
import numpy as np
import torchvision
table = {}
helps = 0
def cache(f):
    def helper(*args):
        global table
        if args in table:
            return table[args]
        print("cache miss")
        res = f(*args)
        table[args] = res
        return res
    return helper

@functools.cache
def get_img(img_path):
    return read_image(img_path, mode=torchvision.io.image.ImageReadMode.RGB) / 255.0

class StrandsImgPairDataset(Dataset):

    def __init__(self, training_input=None, crop_width=56, training=False):
        super(StrandsImgPairDataset, self).__init__()
        self.width = 512
        self.height = 404
        self.crop_width = crop_width
        self.train = training
        self.large_gpu = True
        self.training_input = training_input
        ## training_input = file, file, displacement, feature count
        if not self.train:
            self.data = []
            if self.large_gpu:
                self.image_paths = []
                self.images = []
                idx = 0
                for i, pair in enumerate(self.training_input):
                    path1 = pair[0]
                    path2 = pair[1]
                    if path1 not in self.image_paths:
                        self.image_paths.append(path1)
                        self.images.append(read_image(path1, mode=torchvision.io.image.ImageReadMode.RGB) / 255.0)
                        idx += 1
                    if path2 not in self.image_paths:
                        self.image_paths.append(path2)
                        self.images.append(read_image(path2, mode=torchvision.io.image.ImageReadMode.RGB) / 255.0)
                        idx += 1
                    self.data.append((self.image_paths.index(path1), self.image_paths.index(path2), 0, i))

            else:
                for i, pair in enumerate(self.training_input):
                    # if i == 44:
                    # print(self.GT[i])
                    path1 = pair[0]
                    path2 = pair[1]
                    self.data.append((path1, path2, i))

        else:

            temp = self.training_input[:, 2].astype(np.float32) * self.width
            self.disp = temp.astype(int)
            fcount1 = self.training_input[:, 3].astype(np.float32).astype(np.int32)
            fcount2 = self.training_input[:, 4].astype(np.float32).astype(np.int32)
            fcount_threshold = np.percentile([fcount1, fcount2], 50)
            fcount_threshold = max(fcount_threshold, 250)
            ##print (GT[0])
            qualifieds = np.array(fcount1) >= fcount_threshold
            qualifieds2 = np.array(fcount2) >= fcount_threshold
            qualifieds3 = abs(self.disp) < int(self.width - self.crop_width)
            self.nonzeros = np.count_nonzero(np.logical_and(qualifieds, qualifieds2, qualifieds3))
            print("[+] {} images qualified with t:{}  out of {}, f1: {}, f2: {}, shift > 0.5: {} ".format(
                self.nonzeros, fcount_threshold, len(qualifieds), np.count_nonzero(qualifieds),
                np.count_nonzero(qualifieds2), np.count_nonzero(qualifieds3)))
            if self.nonzeros == 0:
                raise ValueError("[-] no valid selection to teach on")

            self.data = []

            if self.large_gpu:
                self.image_paths = []
                self.images = []
                idx = 0
                for i, pair in enumerate(self.training_input):
                    if qualifieds[i] and qualifieds2[i] and qualifieds3[i]:
                        path1 = pair[0]
                        path2 = pair[1]
                        if path1 not in self.image_paths:
                            self.image_paths.append(path1)
                            self.images.append(read_image(path1, mode=torchvision.io.image.ImageReadMode.RGB) / 255.0)
                            idx += 1
                        if path2 not in self.image_paths:
                            self.image_paths.append(path2)
                            self.images.append(read_image(path2, mode=torchvision.io.image.ImageReadMode.RGB) / 255.0)
                            idx += 1
                        self.data.append(
                            (self.image_paths.index(path1), self.image_paths.index(path2), self.disp[i], i))
            else:
                for i, pair in enumerate(self.training_input):
                    # if i == 44:
                    # print(self.GT[i])
                    path1 = pair[0]
                    path2 = pair[1]
                    if qualifieds[i] and qualifieds2[i] and qualifieds3[i]:
                        self.data.append((path1, path2, self.disp[i], i))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.large_gpu:
            source_img = self.images[self.data[idx][0]]
            target_img = self.images[self.data[idx][1]]
        else:
            source_img = get_img(self.data[idx][0])
            target_img = get_img(self.data[idx][1])
        displ = self.data[idx][2]
        return source_img, target_img, displ


class Strands(StrandsImgPairDataset):
    def __init__(self, crop_width, fraction, smoothness, training_input, training=False, device=t.device("cpu")):
        super().__init__(training_input=training_input, crop_width=crop_width, training=training)
        self.device = device
        self.fraction = fraction
        self.smoothness = smoothness
        # self.flip = K.Hflip()
        self.flip = K.geometry.transform.Hflip()
        print("[+] Strands dataset created")

    #def __getitem__(self, idx):
    #    return self.get_item(idx)
    @functools.cache
    def get_item_cached(self,idx):
        source, target, displ = super().__getitem__(idx)
        # source[:, :32, -64:] = (t.randn((3, 32, 64)) / 4 + 0.5).clip(0.2, 0.8) # for vlurring out the water mark
        # displ = displ*(512.0/self.width)
        cropped_target, crop_start, original_image, blacked_img = crop_img(self.width, self.crop_width, target,
                                                                           displac=displ)
        if self.smoothness == 0:
            heatmap = get_heatmap(self.width, self.fraction, self.crop_width, crop_start)
        else:
            heatmap = get_smooth_heatmap(self.smoothness, self.width, crop_start, self.fraction, self.crop_width)
        # plot_heatmap(source, target, cropped_target, displ, heatmap)
        return source.to(self.device), cropped_target.to(self.device), displ, heatmap.to(self.device)

    def __getitem__(self, idx):
        if self.train:
            return self.get_item_cached(idx)
        else:
            # croping target when evalution is up to 504 pixels
            source, target, displ = super().__getitem__(idx)
            #left = (self.width - self.crop_width) / 2
            #right = (self.width - self.crop_width) / 2 + self.crop_width
            return source.to(self.device), target.to(self.device), displ


def get_heatmap(width, fraction, crop_width, crop_start):
    frac = width // fraction
    heatmap = t.zeros(frac).long()
    idx = int((crop_start + crop_width // 2) * (frac / width))
    heatmap[idx] = 1
    heatmap[idx + 1] = 1
    return heatmap


def get_smooth_heatmap(smoothness, width, crop_start, fraction, crop_width):
    surround = smoothness * 2
    frac = width // fraction
    heatmap = t.zeros(frac + surround)
    idx = int((crop_start + crop_width // 2) * (frac / width)) + smoothness
    heatmap[idx] = 1
    idxs = np.array([-1, +1])
    for i in range(1, smoothness + 1):
        indexes = list(idx + i * idxs)
        for j in indexes:
            if 0 <= j < heatmap.size(0):
                heatmap[j] = 1 - i * (1 / (smoothness + 1))
    return heatmap[surround // 2:-surround // 2]


def crop_img(width, crop_width, img, displac):
    # crop - avoid asking for unavailable crop
    if displac >= 0:
        crops = [random.randint(0, int(width - crop_width - 1) - displac)]
    else:
        crops = [random.randint(0 - displac, int(width - crop_width - 1))]

    crop_start = random.choice(crops)
    crop_out = crop_start + displac
    # crop_start = random.randint(0, self.width - self.crop_width - 1)
    blacked_image = img.clone()
    blacked_image[:, :, crop_start + crop_width:] = 0
    blacked_image[:, :, :crop_start] = 0
    return img[:, :, crop_start:crop_start + crop_width], crop_out, img, blacked_image


if __name__ == '__main__':
    data = StrandsImgPairDataset()
    print(len(data))
    # plot_img_pair(*data[100])
