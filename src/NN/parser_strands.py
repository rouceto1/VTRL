from tkinter import image_names
import torch as t
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.io import read_image, decode_image
import random
import itertools
from glob import glob
import kornia as K
import numpy as np
import torchvision
import pandas as pd
from matplotlib.pyplot import imshow
import statistics

class StrandsImgPairDataset(Dataset):

    def __init__(self, GT,thr):
        super(StrandsImgPairDataset, self).__init__()
        self.width = 760
        self.height = 600
        self.quality_threshold = thr
        self.GT = GT
        self.disp = GT[:,2].astype(np.float32)
        self.fcount = GT[:,3].astype(np.float32).astype(np.int32)
        # GT in format imagea | imageb | displacemetn | feature count | 63x histgram bin|
        print (GT[0])
        qualifieds= np.array(self.fcount) >= max(self.fcount) * 0.1 ## TODO the 0.1 as a paratmeters .. arashes hardoced shit
        #print(qualifieds)
        qualifieds2 = self.disp < 9999
        #print(qualifieds2)
        self.nonzeros=np.count_nonzero(qualifieds)  + np.count_nonzero(qualifieds2)
        print("[+] {} images were qualified out of {} images with threshold {}".format(self.nonzeros,len(qualifieds),self.quality_threshold))
        if self.nonzeros==0:
            print("[-] no valid selection to teach on. Exiting")
            exit(0)

        self.data = []
        for i, pair in enumerate(self.GT):
            path1 = pair[0]
            path2 = pair[1]
            if qualifieds[i] and qualifieds2[i]:
                self.data.append((path1,path2, self.disp[i],i))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        source_img = read_image(self.data[idx][0])/255.0
        target_img = read_image(self.data[idx][1])/255.0
        if not self.GT is None:
            displ=self.data[idx][2]
            return source_img, target_img,displ,self.data[idx][3]
        else:
            return source_img, target_img

class Strands(StrandsImgPairDataset):
    #crop 64
    #fraction 8
    #smoothnes 3
    #datapath path to the main folder???
    # dsp path to the file with displacements
    # seasosns array of names of the folders with iamfes.
    def __init__(self, crop_width, fraction, smoothness, GT,thre=0.25):
        super().__init__(GT=GT, thr=thre)
        self.crop_width = crop_width
        self.fraction = fraction
        self.smoothness = smoothness
        # self.flip = K.Hflip()
        self.flip =K.geometry.transform.Hflip()

    def __getitem__(self, idx):
        if not self.GT is None:
            source, target , displ , data_idx = super().__getitem__(idx)
            #source[:, :32, -64:] = (t.randn((3, 32, 64)) / 4 + 0.5).clip(0.2, 0.8) # for vlurring out the water mark
            cropped_target, crop_start = self.crop_img(target,displ)
            if self.smoothness == 0:
                heatmap = self.get_heatmap(crop_start)
            else:
                heatmap = self.get_smooth_heatmap(crop_start)
            return source, cropped_target, heatmap , data_idx
        else:
            source, target  = super().__getitem__(idx)
            return source, target


    def crop_img(self, img,dspl=0):

        lower_bound = self.crop_width
        upper_bound = self.width-self.crop_width/2
        
        if dspl>0:
            upper_bound = int(upper_bound - dspl)
        elif dspl<0:
            lower_bound = int(lower_bound - dspl)
        #print("u  " , upper_bound , lower_bound, dspl)
        crop_center = random.randint(lower_bound, upper_bound)
        crop_start = crop_center - self.crop_width
        return img[:, :, crop_start:crop_start + self.crop_width], crop_start

    def get_heatmap(self, crop_start):
        frac = self.width // self.fraction ##wivision without reminder 
        heatmap = t.zeros(frac).long()
        idx = int((crop_start + self.crop_width//2) * (frac/self.width))
        heatmap[idx] = 1
        heatmap[idx + 1] = 1
        return heatmap

    def get_smooth_heatmap(self, crop_start):
        surround = self.smoothness * 2
        frac = self.width // self.fraction
        heatmap = t.zeros(frac + surround)
        idx = int((crop_start + self.crop_width//2) * (frac/self.width)) + self.smoothness
        heatmap[idx] = 1
        idxs = np.array([-1, +1])
        for i in range(1, self.smoothness + 1):
            indexes = list(idx + i * idxs)
            for j in indexes:
                if 0 <= j < heatmap.size(0):
                    heatmap[j] = 1 - i * (1/(self.smoothness + 1))
        return heatmap[surround//2:-surround//2]

    def augment(self, source, target):
        # crop the logo - this for some reason makes the network diverge on evaluation set
        # source = source[:, 30:, :]
        # target = target[:, 30:, :]
        source[:, :32, -64:] = (t.randn((3, 32, 64)) / 4 + 0.5).clip(0.2, 0.8)
        if random.random() > 0.95:
            target = source.clone()
        if random.random() > 0.5:
            source = self.flip(source)
            target = self.flip(target)
        return source.squeeze(0), target

if __name__ == '__main__':
    data = StrandsImgPairDataset()
    print(len(data))
    plot_img_pair(*data[100])
