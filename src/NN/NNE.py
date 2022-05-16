#!
import pdb
from tkinter import W
from pkg_resources import evaluate_marker
import torch
import torch as t
from model import Siamese, load_model, get_custom_CNN, save_model
from torch.utils.data import DataLoader
from parser_grief import ImgPairDataset
from parser_nordland import RectifiedNordland 
from torchvision.transforms import Resize
from tqdm import tqdm
from utils import get_shift, plot_samples, plot_displacement, affine
import numpy as np
from scipy import interpolate
import os
from torch.optim import AdamW
from matplotlib.pyplot import imshow
from time import time,ctime
import argparse

parser = argparse.ArgumentParser(description='example: -in_model_path ./results_siam/model.pt --data_path ../grief_jpg/stromovka/ --d0 fall --d1 winter --ocsv ./results_siam/x.csv')
parser.add_argument('--in_model_path', type=str, help="path to the model to be train. if passed string start , it will start with an empty model")
parser.add_argument('--data_path', type=str, help="path to the main data")
parser.add_argument('--d0', type=str, help="path to first dataset")
parser.add_argument('--d1', type=str, help="path to second dataset")
parser.add_argument('--ocsv', type=str, help="output csv file path")

args = parser.parse_args()
in_model_path=args.in_model_path
d0=args.d0
d1=args.d1
ocsv=args.ocsv
data_path=args.data_path


device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
print("[+] device is {}".format(device))
code_run_time=ctime(time()).replace(":","_").replace(" ","_")
VISUALIZE = False
PLOT_IMPORTANCES = False
WIDTH = 512 #! nord width is 512 but stromovka width is 1024
# CROP_SIZE = WIDTH//16 
CROP_SIZE = WIDTH - 8
# CROP_SIZE = 56
SMOOTHNESS = 3
center_mask=0
PAD = (CROP_SIZE - 8) // 16
FRACTION = 8
OUTPUT_SIZE = 64  #  WIDTH // FRACTION
CROPS_MULTIPLIER = 1
BATCHING = CROPS_MULTIPLIER    # this improves evaluation speed by a lot
# LAYER_POOL = True
LAYER_POOL = False
# FILTER_SIZE = 3 #! incompatible with train_siam.py
FILTER_SIZE =5 
END_BN = True
EMB_CHANNELS = 256

EVAL_LIMIT = 2000
TOLERANCE = 48

if "stromovka" in data_path or "carlevaris" in data_path:
    transform = Resize(192)
    # transform = Resize(192 * 2)
    # transform = Resize((288, 512))

def get_histogram(src, tgt):
    #! if for loop below runs, it will try to crop the target from different parts. definitly sth that we dont want
    # target_crops = []
    # for crop_idx in crops_idx:
    #     target_crops.append(tgt[..., crop_idx:crop_idx + CROP_SIZE])
    # target_crops = t.cat(target_crops, dim=0)
    # batched_source = src.repeat(crops_num // BATCHING, 1, 1, 1)
    # histogram = model(batched_source, target_crops, padding=PAD)  # , fourrier=True)
    # histogram = model(batched_source, tgt, padding=PAD)  # , fourrier=True)
    histogram = model(src, tgt, padding=PAD)  # , fourrier=True)
    std, mean = t.std_mean(histogram, dim=-1, keepdim=True)
    histogram = (histogram - mean) / std
    histogram = t.softmax(histogram , dim=1)
    return histogram


def get_importance(src, tgt, displac):
    # displac here is in size of embedding (OUTPUT_SIZE)
    histogram = model(src, tgt, padding=PAD, displac=displac).cpu().numpy()
    f = interpolate.interp1d(np.linspace(0, 512, OUTPUT_SIZE), histogram, kind="cubic")
    interpolated = f(np.arange(512))
    return interpolated[0]


def eval_displacement(eval_model):
    global model
    model = eval_model
    if "nordland" in data_path:
        dataset = RectifiedNordland (CROP_SIZE,FRACTION, SMOOTHNESS ,data_path,None, [d0,d1] ,center_mask)
    elif "grief_jpg" in data_path:
        dataset =ImgPairDataset(data_path)
    train_loader = DataLoader(dataset, 1, shuffle=False)
    model.eval()
    with torch.no_grad():
        abs_err = 0
        valid = 0
        idx = 0
        errors = []
        results = []
        results_gt = []
        if "nordland" in data_path: 
            histograms = np.zeros((14124, 63))
            #! hist rows with zero rows means that they are discarded from evaluation
        elif "stromovka" in data_path:
            histograms = np.zeros((500, 63))
        elif "carlevaris" in data_path:
            histograms = np.zeros((539, 63))
        for batch in tqdm(train_loader):
            if "grief" in data_path:
                source, target , displ = transform(batch[0].to(device)), transform(batch[1].to(device)) , batch[2]
            else:
                source, target = batch[0].to(device) , batch[1].to(device)
            #source, target = batch[0].to(device), batch[1].to(device)
            #! imshow(batch[1][0].numpy().T) # for curiosity
            # do it in both directions target -> source and source -> target
            histogram = get_histogram(source, target)
            histograms[idx, :] = histogram.cpu().numpy()
            shift_hist = histogram.cpu()
            # pdb.set_trace()
            f = interpolate.interp1d(np.linspace(0, 1024, OUTPUT_SIZE-1), shift_hist, kind="cubic") #? why this is needed?
            interpolated = f(np.arange(1024))
            ret = -(np.argmax(interpolated) - 512)
            results.append(ret)
            displac_mult = 1024/WIDTH #? what is this?

            # plot_displacement(source.squeeze(0).cpu(),
            #         target.squeeze(0).cpu(),
            #         shift_hist.squeeze(0).cpu(),
            #         displacement=-displ.detach().numpy()/displac_mult,
            #         name=str(idx),
            #         dir=result_base+"/results_" + MODEL_TYPE + "/eval/")
            #         # dir="results_" + MODEL_TYPE + "/eval_" + MODEL + "/")
            # print("[+] im path is {}".format(result_base+"/results_" + MODEL_TYPE + "/eval"))
            idx += 1
        print("Evaluated: Absolute mean error: {} Predictions in tolerance: {} %".format(abs_err/idx, valid*100/idx))
        np.savetxt(ocsv, histograms, delimiter=",")
        return abs_err/idx, valid*100/idx

if __name__ == '__main__':
    backbone = get_custom_CNN(LAYER_POOL, FILTER_SIZE, EMB_CHANNELS)
    model = Siamese(backbone, padding=PAD, eb=END_BN).to(device)
    model=load_model(model,in_model_path)
    eval_displacement(model) #! commented out just for understanding code
    #print("love you bye!")
