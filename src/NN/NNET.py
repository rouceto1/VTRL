#!
import pdb
from tkinter import W
from pkg_resources import evaluate_marker
import torch
import torch as t
from .model import Siamese, load_model, get_custom_CNN, save_model
from torch.utils.data import DataLoader
from .parser_grief import ImgPairDataset
from .parser_nordland import RectifiedNordland 
from .parser_strands import Strands
from torchvision.transforms import Resize
from tqdm import tqdm
from .utils import get_shift, plot_samples, plot_displacement, affine
import numpy as np
from scipy import interpolate
import os
from torch.optim import AdamW
from matplotlib.pyplot import imshow
from time import time,ctime
from matplotlib import pyplot as plt

from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
BATCH_SIZE= 30
LR = 1e-4
EPOCHS=1
EVAL_RATE = 1
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
print("[+] device is {}".format(device))
code_run_time=ctime(time()).replace(":","_").replace(" ","_")
VISUALIZE = True
PLOT_IMPORTANCES = False
WIDTH = 512 #! nord width is 512 but stromovka width is 1024
# CROP_SIZE = WIDTH//16 
CROP_SIZE2 = WIDTH - 8
CROP_SIZE = 56
SMOOTHNESS = 3
center_mask=0
PAD = (CROP_SIZE - 8) // 16
HISTPAD=(CROP_SIZE2 - 8) // 16
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

TRAINING = 1

#if "stromovka" in data_path or "carlevaris" in data_path:
#    transform = Resize(192)
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
    histogram = model(src, tgt, padding=HISTPAD)  # , fourrier=True)
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


def get_dataset(data_path, GT):
    if "nordland" in data_path: 
        dataset = RectifiedNordland(CROP_SIZE,FRACTION, SMOOTHNESS ,data_path,dsp,[d0,d1],threshold=THRESHOLD)
        histograms = np.zeros((14124, 63))
    elif "stromovka" in data_path:
        histograms = np.zeros((500, 63))
    elif "carlevaris" in data_path:
        histograms = np.zeros((539, 63))
    elif "strand" in data_path:
        if TRAINING > 0:
            crp = CROP_SIZE
        else:
            crp = CROP_SIZE2
        dataset = Strands(crp,FRACTION, SMOOTHNESS ,GT,thre=TRAINING) #TODO: threshold does not work abd is currently 
        histograms = np.zeros((len(GT),63))
    return dataset, histograms

def eval_displacement(eval_model,data_path, GT):
    global model
    model = eval_model
    dataset, histograms = get_dataset(data_path,GT)
    train_loader = DataLoader(dataset, 1, shuffle=False)
    model.eval()
    with torch.no_grad():
        abs_err = 0
        valid = 0
        idx = 0
        errors = []
        results = []
        results_gt = []
        for batch in tqdm(train_loader):
            if "grief" in data_path:
                source, target , displ = transform(batch[0].to(device)), transform(batch[1].to(device)) , batch[2]
            else:
                source, target = batch[0].to(device) , batch[1].to(device)
            #source, target = batch[0].to(device), batch[1].to(device)
            imshow(batch[1][0].numpy().T) # for curiosity
            #plt.show()
            # do it in both directions target -> source and source -> target
            histogram = get_histogram(source, target)
            histograms[idx, :] = histogram.cpu().numpy()
            shift_hist = histogram.cpu()
            # pdb.set_trace()
            f = interpolate.interp1d(np.linspace(0, WIDTH, OUTPUT_SIZE-1), shift_hist, kind="cubic") #? why this is needed?
            interpolated = f(np.arange(WIDTH))
            ret = -(np.argmax(interpolated) - WIDTH/2)
            results.append(ret)
            displac_mult = WIDTH/WIDTH #? what is this?
            print(source.squeeze(0).cpu().size())
           # plt.imshow(np.rot90(batch[0][0].numpy().T),origin='lower')
            #plt.show()
            plot_displacement(source.squeeze(0).cpu(),
                    target.squeeze(0).cpu(),
                    shift_hist.squeeze(0).cpu(),
                    displacement = ret,
                    name=str(idx))
                    # dir="results_" + MODEL_TYPE + "/eval_" + MODEL + "/")
            idx += 1
        print("Evaluated: Absolute mean error: {} Predictions in tolerance: {} %".format(abs_err/idx, valid*100/idx))
        #np.savetxt(ocsv, histograms, delimiter=",")
        return abs_err/idx, valid*100/idx, histograms, ret

def train_loop(epoch, GT = 0, data_path = "", ):
    global PAD , model,optimizer ,loss
    NEGATIVE_FRAC = 1/3
    model.train()
    loss_sum = 0
    generation = 0
    #dataset = RectifiedNordland(CROP_SIZE,FRACTION, SMOOTHNESS ,data_path,dsp,[d0,d1],threshold=THRESHOLD) #!
    dataset, histograms = get_dataset(data_path,GT)
    train_loader = DataLoader(dataset, BATCH_SIZE, shuffle=False)
    idx=0
    #try:
    if 1: 
        for batch in tqdm(train_loader):
            source, target, heatmap = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            #! plt.imshow(batch[0][0].numpy().T) # for curiosity
            # source = batch_augmentations(source) #! no augmentation with gt
            #! if NEGATIVE_FRAC > 0.01:
            #! batch, heatmap = hard_negatives(source, heatmap)
            out = model(source, target, padding=PAD)
            #print(source, target)
            optimizer.zero_grad()
            l = loss(out, heatmap)
            loss_sum += l.cpu().detach().numpy()
            l.backward()
            optimizer.step()
            idx+=1
            if True:
                plot_samples(source[0].cpu(),
                             target[0].cpu(),
                             heatmap[0].cpu(),
                             prediction=out[0].cpu(),
                             name=str(idx)
                             )
    #except Exception as e:
    #    print ("TRAINING FAILED")
    #    print(e)
    print("Training of epoch", epoch, "ended with loss", loss_sum / len(train_loader))

def NNeval_from_python(files, data_path, weights_file):
    global backbone, model, optimizer, loss 
    global TRAINING
    TRAINING = -1
    backbone = get_custom_CNN(LAYER_POOL, FILTER_SIZE, EMB_CHANNELS)
    model = Siamese(backbone, padding=HISTPAD, eb=END_BN).to(device)
    model=load_model(model,weights_file)
    return eval_displacement(model,data_path,files) #! commented out just for understanding code

def NNteach_from_python(GT, data_path, weights_file, epochs):
    global backbone, model, optimizer, loss
    global TRAINING 
    TRAINING = 1
    backbone = get_custom_CNN(LAYER_POOL, FILTER_SIZE, EMB_CHANNELS)
    model = Siamese(backbone, padding=PAD, eb=END_BN).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    loss = BCEWithLogitsLoss()
    for epoch in range(EPOCHS):
        train_loop(epoch, GT, data_path)
        save_model(model, optimizer,weights_file,epoch) #!

if __name__ == '__main__':
    backbone = get_custom_CNN(LAYER_POOL, FILTER_SIZE, EMB_CHANNELS)
    model = Siamese(backbone, padding=PAD, eb=END_BN).to(device)
    model=load_model(model,in_model_path)
    eval_displacement(model) #! commented out just for understanding code
