#!/usr/bin/env python3.9
import copy
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from .evaluate_model import eval_displacement
from .utils import batch_augmentations, plot_samples, plot_similarity, plot_displacement, plot_heatmap, plot_histogram
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


def train_loop(epoch, model, train_loader, optimizer, out_folder):
    global batch_aug
    model.train()
    loss_sum = 0
    count = 0
    print("Training model epoch", epoch)
    for batch in tqdm(train_loader):
        source, target, heatmap, u_target = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[4].to(device)
        source = batch_aug(source)
        count = count + 1
        if conf["plot_training"]:
            if count < 10:
                blacked = batch[6].to(device)
                plot_heatmap(source[0].cpu(),
                             u_target[0].cpu(),
                             heatmap=heatmap[0].cpu(),
                             cropped_target=target[0].cpu(),
                             blacked_image=blacked[0].cpu(),
                             name=str(epoch) + "_" + str(count),
                             dir=out_folder)
        if conf["negative_frac"] > 0.01:
            batch, heatmap = hard_negatives(source, heatmap)
        out = model(source, target, padding=conf["pad"])

        optimizer.zero_grad()

        los = loss(out, heatmap)
        loss_sum += los.cpu().detach().numpy()
        los.backward()

        optimizer.step()


    print("Training of epoch", epoch, "ended with loss", loss_sum / len(train_loader))
    return model



def eval_loop(val_loader, model, epoch, histograms,out_folder):
    global conf
    model.eval()
    with t.no_grad():
        mae, acc, hist, disp = eval_displacement(eval_model=model, loader=val_loader,
                                                 histograms=np.zeros((len(val_loader), 64)), conf=conf,
                                                 padding=conf["pad_teach"])
        count = 0
        if conf["plot_eval"]:
            for batch in val_loader:
                pass
                if count > 5:
                    break
                source, target = (batch[0].to(device)), (batch[4].to(device))
                plot_histogram(source, target, displacement=disp[count], histogram=hist[count],
                               name=str(epoch) + "_" + str(count),
                                dir=out_folder)
                count = count + 1
    print("Eval of epoch " + str(epoch) + " ended with mae " + str(mae))
    return mae


def teach_stuff(train_data, data_path, eval_model=None, out=None, model_path=None):
    lowest_err = 9999999
    global model
    global conf
    best_model = None
    model, conf = get_model(model, model_path, eval_model, conf, conf["pad_teach"])
    optimizer = AdamW(model.parameters(), lr=10**-conf["lr"]) #conf["lr"])

    dataset, histograms = get_dataset(data_path, train_data, conf, training=True)
    train_size = int(len(dataset) * 0.95)
    val, train = t.utils.data.random_split(dataset, [len(dataset) - train_size, train_size])
    train_loader = DataLoader(train, conf["batch_size_train"], shuffle=True,  num_workers=10)
    val_loader = DataLoader(val, conf["batch_size_eval"], shuffle=False,  num_workers=10)
    if conf["epochs"] % conf["eval_rate"] != 0:
        print("WARNING epochs and eval rate are not divisible")
    for epoch in range(conf["epochs"]):
        if epoch % conf["eval_rate"] == 0:  # and epoch > 0:
            err = eval_loop(val_loader, model, epoch, histograms,out)
            if err < lowest_err:
                lowest_err = err
                best_model = copy.deepcopy(model)
        model = train_loop(epoch, model, train_loader, optimizer, out)

    save_model_to_file(best_model, model_path, lowest_err, optimizer)


def NNteach_from_python(training_data, data_path, experiments_path, config):
    global conf
    global device
    conf = config
    device = conf["device"]
    global batch_aug
    batch_aug = batch_augmentations.to(device)
    print("trianing:" + str(experiments_path))
    teach_stuff(train_data=training_data, model_path=os.path.join(experiments_path, "weights.pt"), out=experiments_path,
                data_path=data_path)


if __name__ == '__main__':
    pass
    # teach_stuff()
