#!/usr/bin/env python3.9
import numpy as np
import torch
import os
import torch as t
from scipy import interpolate
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from tqdm import tqdm

from .model import load_model, get_parametrized_model, save_model_to_file
from .utils import plot_displacement
from ..NN.parser_strands import Strands
import yaml
from pathlib import Path


def get_pad(crop):
    return (crop - 8) // 16


def load_config(conf_path):
    conf = yaml.safe_load(Path(conf_path).read_text())
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
    conf["device"] = device
    output_size = conf["width"] // conf["fraction"]
    MASK = t.zeros(output_size)
    PAD = get_pad(conf["crop_sizes"][0])
    MASK[:PAD] = t.flip(t.arange(0, PAD), dims=[0])
    MASK[-PAD:] = t.arange(0, PAD)
    MASK = output_size - 1 - MASK
    MASK = (output_size - 1) / MASK.to(device)
    conf["mask"] = MASK
    conf["pad"] = PAD
    conf["output_size"] = output_size
    conf["crop_size"] = conf["width"] - 8
    conf["histpad"] = (conf["crop_size"] - 8) // 16
    conf["batching"] = conf["crops_multiplier"]
    return conf


def get_dataset(data_path, GT, conf):
    if "nordland" in data_path:
        dataset = RectifiedNordland(conf["crop_size"], conf["fraction"], conf["smoothness"], data_path, dsp, [d0, d1],
                                    threshold=conf["threshold"])
        histograms = np.zeros((14124, 63))
    elif "stromovka" in data_path:
        histograms = np.zeros((500, 63))
    elif "carlevaris" in data_path:
        histograms = np.zeros((539, 63))
    elif "strand" in data_path:
        crp = conf["crop_size"]
        dataset = Strands(crp, conf["fraction"], conf["smoothness"], GT,
                          thre=-1)  # TODO: threshold does not work abd is currently
        histograms = np.zeros((len(GT), 63))
    return dataset, histograms


def get_model(model, model_path, eval_model, conf):
    if eval_model is not None:
        model = eval_model
        return model, conf

    if "tiny" in model_path:
        conf["emb_channels"] = 16
        use256 = False
    else:
        conf["emb_channels"] = 256
        use256 = True
    model = get_parametrized_model(conf["layer_pool"], conf["filter_size"], conf["emb_channels"], conf["residual"],
                                   conf["pad"], conf["device"], legacy=use256)
    if os.path.exists(model_path):
        model = load_model(model, model_path)
    return model, conf
