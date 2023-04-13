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

class RectifiedImgPairDataset(Dataset):

    def __init__(self, path,GT_path,seasons,threshold):
        super(RectifiedImgPairDataset, self).__init__()
        self.width = 512
        self.height = 288
        # self.quality_threshold = 150
        #self.quality_threshold = 400
        # self.quality_threshold = 200
        self.quality_threshold = threshold
        self.GT_path=GT_path
        valid_subfolders =seasons 
        lvl1_subfolders = [f.path for f in os.scandir(path) if (f.is_dir() and str(f.path).split("/")[-1] in valid_subfolders)]
        all_season_images_pths = {}
        for subfolder in lvl1_subfolders:
            files = glob(subfolder + '/**/*.png', recursive=True)
            all_season_images_pths[subfolder] = files

        perms = []
        stuff =list(range(0,len(seasons))) 
        for L in range(0, len(stuff) + 1):
            for subset in itertools.combinations(stuff, L):
                if len(subset) == 2:
                    perms.append(subset)
        #print("[+] perms {}".format(perms))

        if not self.GT_path is None:
            try:
                GT=np.loadtxt(self.GT_path,delimiter=',',usecols=range(3))
            except:
                GT=np.loadtxt(self.GT_path)
            # qualifieds= GT[:,1]>=self.quality_threshold
            qualifieds= GT[:,1]>=max(GT[:,1])*0.1

            nonzeros=np.count_nonzero(qualifieds)
            print("[+] {} images were qualified out of {} images with threshold {}".format(nonzeros,len(qualifieds),self.quality_threshold))
            if nonzeros==0:
                print("[-] no valid selection")
                exit(0)

        self.data = []
        for pair in perms:
            subfolder1 = lvl1_subfolders[pair[0]]
            subfolder2 = lvl1_subfolders[pair[1]]
            #! GT is in sorted manner so subfolders should be sorted in name as well
            im_names_sorted=sorted(all_season_images_pths[subfolder1])
            if not self.GT_path is None:
                im_names_sorted=im_names_sorted[0:len(qualifieds)]
            for filepath in im_names_sorted:
                fileindex = filepath.split("/")[-1][:-4]
                if not self.GT_path is None:
                    if qualifieds[int(fileindex)]:
                        pair = (os.path.join(subfolder1, fileindex + ".png"), os.path.join(subfolder2, fileindex + ".png"),GT[int(fileindex),0],int(fileindex))
                        self.data.append(pair)
                else:
                        pair = (os.path.join(subfolder1, fileindex + ".png"), os.path.join(subfolder2, fileindex + ".png"))
                        self.data.append(pair)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #! theses lines are for swaping src and trgt and src by prob of 0.5
        # if random.random() > 0.0:
        #     a, b = 0, 1
        # else:
        #     b, a = 0, 1
        # a, b = 0, 1 #? not sure if this uswaped or swapped #! this one was wrong
        a, b = 0, 1 #? not sure if this uswaped or swapped

        source_img = read_image(self.data[idx][a])/255.0
        target_img = read_image(self.data[idx][b])/255.0
        if not self.GT_path is None:
            displ=self.data[idx][2]
            return source_img, target_img,displ,self.data[idx][3]
        else:
            return source_img, target_img
class RectifiedNordland(RectifiedImgPairDataset):

    def __init__(self, crop_width, fraction, smoothness, data_path,dsp,seasons,center_mask=48,threshold=0.25):
        super(RectifiedNordland, self).__init__(path=data_path,GT_path=dsp,seasons=seasons,threshold=threshold)
        self.crop_width = crop_width
        self.fraction = fraction
        self.smoothness = smoothness
        self.center_mask = center_mask 
        # self.flip = K.Hflip()
        self.flip =K.geometry.transform.Hflip()

    def __getitem__(self, idx):
        if not self.GT_path is None:
            source, target , displ , data_idx = super(RectifiedNordland, self).__getitem__(idx)
            source[:, :32, -64:] = (t.randn((3, 32, 64)) / 4 + 0.5).clip(0.2, 0.8) # for vlurring out the water mark
            cropped_target, crop_start = self.crop_img(target,displ)
            if self.smoothness == 0:
                heatmap = self.get_heatmap(crop_start)
            else:
                heatmap = self.get_smooth_heatmap(crop_start)
            return source, cropped_target, heatmap , data_idx
        else:
            source, target  = super(RectifiedNordland, self).__getitem__(idx)
            return source, target

    def set_crop_size(self, crop_size, smoothness=None):
        self.crop_width = crop_size
        if smoothness is None:
            self.smoothness = (crop_size // 8) - 1
        else:
            self.smoothness = smoothness

    def crop_img(self, img,dspl=0):
        #! crop - avoid center (rails) and edges
        if dspl>0:
            dspl=int(abs(dspl))
            ranges=[[0,self.width//2-self.center_mask-self.crop_width],[self.width//2+self.center_mask,self.width-self.crop_width-dspl-1]]
        elif dspl<0:
            dspl=int(abs(dspl))
            ranges=[ [dspl,self.width//2-self.center_mask-self.crop_width] , [self.width//2+self.center_mask,self.width-self.crop_width-1] ]
        else:
            ranges=[ [0, int(self.width / 2 - self.center_mask - self.crop_width)],[int(self.width / 2 + self.center_mask), int(self.width - self.crop_width - 1)]]
        if ranges[0][0]<0 or ranges[0][1]<0 or ranges[1][0]<0 or ranges[1][1]<0:
            print("[-] range selection is not possibile {}".format(ranges))
            exit(1)
        if ranges[0][0]< ranges[0][1] and ranges[1][0] < ranges[1][1]:
            crops = [random.randint(ranges[0][0], ranges[0][1]), random.randint(ranges[1][0],ranges[1][1])]
            crop_start = random.choice(crops)
        elif ranges[0][0]>= ranges[0][1] and ranges[1][0]< ranges[1][1]:
            crop_start = random.randint(ranges[1][0],ranges[1][1])
        elif ranges[0][0]< ranges[0][1] and ranges[1][0]>= ranges[1][1]:
            crop_start = random.randint(ranges[0][0],ranges[0][1])
        else:
            raise NameError("[-] crop is not possible ranges {}".format(ranges))
        return img[:, :, crop_start:crop_start + self.crop_width], crop_start

    def get_heatmap(self, crop_start):
        frac = self.width // self.fraction
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
    data = RectifiedImgPairDataset()
    print(len(data))
    plot_img_pair(*data[100])
