#!/usr/bin/env python3.9

import copy

from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .evaluate_model import eval_displacement
# from .parser_nordland import RectifiedNordland
from .utils import plot_samples, batch_augmentations

from .train_eval_common import *

VISUALISE = True
WANDB = False
conf = load_config("./NN_config.yaml")
device = conf["device"]
batch_augmentations = batch_augmentations.to(device)
loss = BCEWithLogitsLoss()
model = None


def hard_negatives(batch, heatmaps):
    if batch.shape[0] == conf["batch_size"] - 1:
        num = int(conf["batch_size"] * conf["negative_frac"])
        if num % 2 == 1:
            num -= 1
        indices = t.tensor(np.random.choice(np.arange(0, conf["batch_size"]), num), device=device)
        heatmaps[indices, :] = 0.0
        tmp_t = t.clone(batch[indices[:num // 2]])
        batch[indices[:num // 2]] = batch[indices[num // 2:]]
        batch[indices[num // 2:]] = tmp_t
        return batch, heatmaps
    else:
        return batch, heatmaps


def train_loop(epoch, train_loader, optimizer):
    global model
    model.train()
    loss_sum = 0
    print("Training model epoch", epoch)
    generation = 0
    for batch in tqdm(train_loader):
        source, target, heatmap = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        source = batch_augmentations(source)
        if conf["negative_frac"] > 0.01:
            batch, heatmap = hard_negatives(source, heatmap)
        out = model(source, target, padding=conf["pad"])
        optimizer.zero_grad()
        l = loss(out, heatmap)
        loss_sum += l.cpu().detach().numpy()
        l.backward()
        optimizer.step()
    print("Training of epoch", epoch, "ended with loss", loss_sum / len(train_loader))


def eval_loop(epoch, val_loader):
    global model
    model.eval()
    with torch.no_grad():
        mae, acc, _, _ = eval_displacement(eval_model=model, loader=val_loader, hist_padding=conf["histpad_eval"])
    return mae


def teach_stuff(train_data, data_path, eval_model=None, model_path=None):
    LOAD_EPOCH = 0
    lowest_err = 9999999
    global model
    global conf
    model, conf = get_model(model, model_path, eval_model, conf)
    optimizer = AdamW(model.parameters(), lr=conf["lr"])

    dataset, histograms = get_dataset(data_path, train_data, conf, training=True)
    val, train = t.utils.data.random_split(dataset, [int(0.05 * len(dataset)), int(0.95 * len(dataset)) + 1])
    train_loader = DataLoader(train, conf["batch_size"], shuffle=True)
    val_loader = DataLoader(val, 1, shuffle=False)
    if conf["epochs"] % conf["eval_rate"] != 0:
        print("WARNING epochs and eval rate are not divisible")
    for epoch in range(LOAD_EPOCH, conf["epochs"]):
        if epoch % conf["eval_rate"] == 0:
            err = eval_loop(epoch, val_loader)
            if err < lowest_err:
                lowest_err = err
                best_model = copy.deepcopy(model)
        train_loop(epoch, train_loader, optimizer)

    save_model_to_file(model, conf["name"], lowest_err, optimizer)


def NNteach_from_python(training_data, data_path, weights_file, epochs):
    global conf
    conf["epochs"] = epochs
    print("trianing:" + str(weights_file))
    teach_stuff(train_data=training_data, model_path=weights_file, data_path=data_path)


if __name__ == '__main__':
    teach_stuff()
