#!/usr/bin/env python3.9

import copy
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from .evaluate_model import eval_displacement
from .utils import batch_augmentations
from .model import save_model_to_file
from .train_eval_common import *
import torch as t

loss = BCEWithLogitsLoss()
model = None


def hard_negatives(batch, heatmaps):
    if batch.shape[0] == conf["batch_size_train"] - 1:
        num = int(conf["batch_size_train"] * conf["negative_frac"])
        if num % 2 == 1:
            num -= 1
        indices = t.tensor(np.random.choice(np.arange(0, conf["batch_size_train"]), num), device=device)
        heatmaps[indices, :] = 0.0
        tmp_t = t.clone(batch[indices[:num // 2]])
        batch[indices[:num // 2]] = batch[indices[num // 2:]]
        batch[indices[num // 2:]] = tmp_t
        return batch, heatmaps
    else:
        return batch, heatmaps


def train_loop(epoch, train_loader, optimizer):
    global model
    global batch_aug
    model.train()
    loss_sum = 0
    print("Training model epoch", epoch)
    for batch in tqdm(train_loader):
        source, target, heatmap = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        source = batch_aug(source)
        if conf["negative_frac"] > 0.01:
            batch, heatmap = hard_negatives(source, heatmap)
        out = model(source, target, padding=conf["pad"])
        optimizer.zero_grad()
        los = loss(out, heatmap)
        loss_sum += los.cpu().detach().numpy()
        los.backward()
        optimizer.step()
    print("Training of epoch", epoch, "ended with loss", loss_sum / len(train_loader))


def eval_loop(val_loader):
    global model
    global conf
    model.eval()
    with t.no_grad():
        mae, acc, _, _ = eval_displacement(eval_model=model, loader=val_loader, conf=conf, padding=conf["pad_teach"])
    return mae


def teach_stuff(train_data, data_path, eval_model=None, model_path=None):
    lowest_err = 9999999
    global model
    global conf
    best_model = None
    model, conf = get_model(model, model_path, eval_model, conf, conf["pad_teach"])
    optimizer = AdamW(model.parameters(), lr=conf["lr"])

    dataset, histograms = get_dataset(data_path, train_data, conf, training=True)
    train_size = int(len(dataset) * 0.95)
    val, train = t.utils.data.random_split(dataset, [len(dataset) - train_size, train_size])
    train_loader = DataLoader(train, conf["batch_size_train"], shuffle=True,pin_memory=True,num_workers=10)
    val_loader = DataLoader(val, conf["batch_size_eval"], shuffle=False,pin_memory=True,num_workers=10)
    if conf["epochs"] % conf["eval_rate"] != 0:
        print("WARNING epochs and eval rate are not divisible")
    for epoch in range(conf["epochs"]):
        if epoch % conf["eval_rate"] == 0:
            err = eval_loop(val_loader)
            if err < lowest_err:
                lowest_err = err
                best_model = copy.deepcopy(model)
        train_loop(epoch, train_loader, optimizer)

    save_model_to_file(best_model, model_path, lowest_err, optimizer)


def NNteach_from_python(training_data, data_path, weights_file, config):
    global conf
    global device
    conf = config
    device = conf["device"]
    global batch_aug
    batch_aug = batch_augmentations.to(device)
    print("trianing:" + str(weights_file))
    teach_stuff(train_data=training_data, model_path=weights_file, data_path=data_path)


if __name__ == '__main__':
    pass
    # teach_stuff()
