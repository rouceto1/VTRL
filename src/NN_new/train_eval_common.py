#!/usr/bin/env python3.9
import numpy as np
import os
import torch as t

from .model import load_model, get_parametrized_model
from ..NN_new.parser_strands import Strands
import yaml
from pathlib import Path


def get_pad(crop):
    return (crop - 8) // 16


def load_config(conf_path, image_width=512, image_height=384):
    conf = yaml.safe_load(Path(conf_path).read_text())
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
    conf["device"] = device
    output_size = conf["width"] // conf["fraction"]
    PAD = get_pad(conf["crop_sizes"][0])
    conf["pad"] = PAD
    conf["output_size"] = output_size
    conf["crop_size_eval"] = conf["width"] - 8
    conf["crop_size_teach"] = conf["crop_sizes"][0]
    conf["pad_eval"] = get_pad(conf["crop_size_eval"])
    conf["pad_teach"] = get_pad(conf["crop_size_teach"])
    conf["batching"] = conf["crops_multiplier"]
    conf["size_frac"] = conf["width"] / image_width
    conf["image_height"] = image_height
    return conf


## training signifies if the data loading is done for training or testing
def get_dataset(data_path, training_input, conf, training=False):
    dataset = None
    histograms = None
    if "nordland" in data_path:
        # dataset = RectifiedNordland(conf["crop_size"], conf["fraction"], conf["smoothness"], data_path, dsp, [d0, d1],
        #                            threshold=conf["threshold"])
        histograms = np.zeros((14124, 63))
    elif "stromovka" in data_path:
        histograms = np.zeros((500, 63))
    elif "carlevaris" in data_path:
        histograms = np.zeros((539, 63))
    elif "strand" in data_path:
        if training:
            crop = conf["crop_size_teach"]
        else:
            crop = conf["crop_size_eval"]

        ## training_input = file, file, [displacement, feature count] (d,f needed when training = True)
        dataset = Strands(crop, conf["fraction"], conf["smoothness"], training_input, training=training)
        # TODO: threshold selects if  iterator returns GT as well (0 > means there is going to be GT)
        histograms = np.zeros((len(training_input), 63))
    return dataset, histograms


def get_model(model, model_path, eval_model, conf, pad):
    if eval_model is not None:
        model = eval_model
        return model, conf

    if "tiny" in model_path:
        conf["emb_channels"] = 16
        # TODO. find all possible changes in NN_config for tiny model and implement
        use256 = False
    else:
        conf["emb_channels"] = 256
        use256 = True
    model = get_parametrized_model(conf["layer_pool"], conf["filter_size"], conf["emb_channels"], conf["residual"],
                                   pad, conf["device"], legacy=use256)
    if os.path.exists(model_path):
        model = load_model(model, model_path)
    return model, conf
