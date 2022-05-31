import torch as t
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.io import read_image, decode_image
import random
from .utils import plot_samples
import pandas as pd
import numpy as np
import json
from matplotlib.pyplot import imshow

class ImgPairDataset(Dataset):

    def __init__(self, path):
        super(ImgPairDataset, self).__init__()
        self.width = 1024
        self.height = 384

        lvl1_subfolders = [path]
        lvl2_subfolders = []
        for subfolder in lvl1_subfolders:
            lvl2_subfolder = [f.path for f in os.scandir(subfolder) if f.is_dir() and "season" in f.name]
            lvl2_subfolders.append(lvl2_subfolder)
        images_displacements = {}
        for subfolder in lvl2_subfolders:
            for subsubfolder in subfolder:
                os.path.join(subsubfolder, "displacements.txt")
                file = open(os.path.join(subsubfolder, "displacements.txt"), 'r')
                displacements = []
                for idx, line in enumerate(file):
                    try:
                        displacements.append(int(line.split(" ")[0]))
                    except:
                        #! it means that data is in float form. because of other codes output
                        displacements.append(int(float(line.split(" ")[0])))
                images_displacements[subsubfolder] = displacements
        season_pairs = []
        for subsubfolder in lvl2_subfolders:
            res = [(a, b) for idx, a in enumerate(subsubfolder) for b in subsubfolder[idx + 1:]]
            season_pairs.extend(res)
        self.annotated_img_pairs = []
        for pair in season_pairs:
            generated_triplets = [(os.path.join(pair[0], str(idx).zfill(9)) + ".jpg",
                                   os.path.join(pair[1], str(idx).zfill(9)) + ".jpg",
                                   images_displacements[pair[1]][idx] - images_displacements[pair[0]][idx])
                                  for idx in range(len(images_displacements[pair[0]]))]
            self.annotated_img_pairs.extend(generated_triplets)

    def __len__(self):
        return len(self.annotated_img_pairs)

    def __getitem__(self, idx):
        #if random.random() > 0.0:
        a, b = 0, 1
        #    displacement = self.annotated_img_pairs[idx][2]
        #else:
        #    b, a = 0, 1
        displacement = -self.annotated_img_pairs[idx][2]

        source_img = read_image(self.annotated_img_pairs[idx][a])/255.0
        target_img = read_image(self.annotated_img_pairs[idx][b])/255.0
        return source_img, target_img, displacement


class CroppedImgPairDataset(ImgPairDataset):

    def __init__(self, crop_width, fraction, smoothness, path="/home/zdeeno/Documents/Datasets/grief_jpg", transforms=None,dataset=None,allign_flag=False):
        super(CroppedImgPairDataset, self).__init__(path=path,dataset=dataset)
        self.crop_width = crop_width
        self.fraction = fraction
        self.smoothness = smoothness
        self.tr = transforms
        self.allign_flag=allign_flag

    def __getitem__(self, idx):
        source, target, displac = super(CroppedImgPairDataset, self).__getitem__(idx)
        displac = displac/2
        if self.tr is not None:
            source = self.tr(source)
            target = self.tr(target)
        if self.allign_flag :
            source, target = self.allign(source, target, displac) 
        cropped_target, crop_start = self.crop_img(target)
        if self.smoothness == 0:
            heatmap = self.get_heatmap(crop_start, displac)
        else:
            heatmap = self.get_smooth_heatmap(crop_start, displac)
        return source, cropped_target, heatmap

    def allign(self,  source, target, displac ):
        dsp=int(displac)
        mid=self.width//2
        if dsp<0: #! not sure about sign here // positive is displacement to right
            source=source[: , : , 0:dsp ]
            target=target[: , : , -dsp: ]
        else:
            source=source[: , : , dsp: ]
            target=target[: , : , 0:-dsp ]
        return source, target

    def crop_img(self, img):
        crop_start = random.randint(self.crop_width, self.width - 2*self.crop_width - 1)
        return img[:, :, crop_start:crop_start + self.crop_width], crop_start

    def get_heatmap(self, crop_start, displacement):
        frac = self.width // self.fraction - 1
        heatmap = t.zeros(frac).long()
        idx = int((crop_start - displacement + self.crop_width//2) / self.fraction)
        if 0 <= idx < 31:
            heatmap[idx] = 1
        return heatmap

    def get_smooth_heatmap(self, crop_start, displacement):
        surround = self.smoothness * 2
        frac = self.width // self.fraction - 1
        heatmap = t.zeros(frac + surround)
        idx = int((crop_start - displacement + self.crop_width//2) / self.fraction) + 3
        idxs = t.tensor([-1, +1])
        for i in range(self.smoothness):
            for j in idx + i*idxs:
                if 0 <= j < heatmap.size(0):
                    heatmap[j] = 1 - i * (1/self.smoothness)
        return heatmap[surround//2:-surround//2]


def test_annotations():
    data_name = "stromovka"
    dataset = ImgPairDataset(dataset=data_name)
    df = pd.read_csv("/home/zdeeno/Downloads/new_annot.csv")
    old_displac = np.zeros(500)
    new_displac = np.zeros(500)
    for pair in dataset.annotated_img_pairs:
        img_idx = int(pair[0].split("/")[-1].split(".")[0])
        old_displac[img_idx] = pair[2]
    for entry in df.iterrows():
        json_str = entry[1]["meta_info"].replace("'", "\"")
        entry_dict = json.loads(json_str)
        if entry_dict["dataset"] == data_name:
            img_idx = int(entry_dict["place"])
            diff = 0
            kp_dict1 = json.loads(entry[1]["kp-1"].replace("'", "\""))
            kp_dict2 = json.loads(entry[1]["kp-2"].replace("'", "\""))
            for kp1, kp2 in zip(kp_dict1, kp_dict2):
                diff += (kp1["x"]/100) * 1024 - (kp2["x"]/100) * 1024
            new_displac[img_idx] = diff // len(kp_dict1)
    old_displac = old_displac[new_displac != 0]
    new_displac = new_displac[new_displac != 0]
    return old_displac, new_displac

class RectifiedEULongterm(ImgPairDataset):

    def __init__(self, crop_width, fraction, smoothness, path=None):
        super(RectifiedEULongterm, self).__init__(path)
        self.crop_width = crop_width
        self.fraction = fraction
        self.smoothness = smoothness
        self.center_mask = 48
        self.flip = K.Hflip()

    def __getitem__(self, idx):
        source, target, displacement = super(RectifiedEULongterm, self).__getitem__(idx)
        # flipping with displacement?
        # source, target = self.augment(source, target)

        cropped_target, crop_start = self.crop_img(target, displacement)
        if self.smoothness == 0:
            heatmap = self.get_heatmap(crop_start)
        else:
            heatmap = self.get_smooth_heatmap(crop_start)
        return source, cropped_target, heatmap

    def set_crop_size(self, crop_size, smoothness=None):
        self.crop_width = crop_size
        if smoothness is None:
            self.smoothness = (crop_size // 8) - 1
        else:
            self.smoothness = smoothness

    def crop_img(self, img, displac):
        # crop - avoid asking for unavailable crop
        if displac >= 0:
            crops = [random.randint(0 + displac, int(self.width - self.crop_width - 1))]
        else:
            crops = [random.randint(0, int(self.width - self.crop_width - 1) + displac)]

        crop_start = random.choice(crops)
        crop_out = crop_start - displac
        # crop_start = random.randint(0, self.width - self.crop_width - 1)
        return img[:, :, crop_start:crop_start + self.crop_width], crop_out

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
    # dataset = CroppedImgPairDataset(64, 8, 3)
    # a, b, c = dataset[0]
    # plot_samples(a, b, c)
    old, new = test_annotations()
