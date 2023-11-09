#!/usr/bin/env python3.9
import copy
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from .evaluate_model import *
from .utils import batch_augmentations, plot_samples, plot_similarity, plot_displacement, plot_heatmap
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

        indices = t.tensor(np.random.choice(np.arange(0, len(heatmaps)), num), device=device)
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
    for batch in train_loader:
        source, target, heatmap = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        source = batch_aug(source)
        count = count + 1

        if conf["negative_frac"] > 0.01:
            batch, heatmap = hard_negatives(source, heatmap)
        out = model(source, target, padding=conf["pad"])

        optimizer.zero_grad()

        los = loss(out, heatmap)
        loss_sum += los.cpu().detach().numpy()
        los.backward()

        optimizer.step()

    return model, loss_sum / len(train_loader)


def eval_loop(val_loader, model, epoch, histograms, out_folder):
    global conf
    model.eval()
    with t.no_grad():
        mae, acc, hist, disp = eval_displacement(eval_model=model, loader=val_loader,
                                                 histograms=np.zeros((len(val_loader), 64)), conf=conf,
                                                 padding=conf["pad_teach"], plot_path=out_folder,
                                                 plot_name="train_hist", epoch=epoch, is_teaching=True)
    return mae.item()


def teach_stuff(train_data, data_path, eval_model=None, out=None, model_path_init=None, model_path_out=None):
    lowest_err = 9999999
    global model
    global conf
    best_model = None
    model, conf = get_model(model, model_path_init, eval_model, conf, conf["pad_teach"])
    optimizer = AdamW(model.parameters(), lr=10 ** -conf["lr"])  # conf["lr"])

    dataset, histograms = get_dataset(data_path, train_data, conf, training=True)
    train_size = int(len(dataset) * 0.95)
    val, train = t.utils.data.random_split(dataset, [len(dataset) - train_size, train_size],)
    train_loader = DataLoader(train, conf["batch_size_train"], shuffle=True, num_workers=10)
    val_loader = DataLoader(val, conf["batch_size_eval"], shuffle=False, num_workers=10)
    if conf["epochs"] % conf["eval_rate"] != 0:
        print("WARNING epochs and eval rate are not divisible")
    losses = []
    meaes = []
    if conf["epochs"] == 0:
        save_model_to_file(model, model_path_out, 0, optimizer)
        return
    for epoch in tqdm(range(1, conf["epochs"] + 1), desc="Training: "):
        if epoch % conf["eval_rate"] == 0 or conf["epochs"] == epoch:  # and epoch > 0:
            err = eval_loop(val_loader, model, epoch, histograms, out)
            meaes.append(err)
            if err < lowest_err:
                lowest_err = err
                best_model = copy.deepcopy(model)
        model, loss = train_loop(epoch, model, train_loader, optimizer, out)
        losses.append(loss)
    #print("Training ended with losses: " + str(losses))
    #print("Training progressed with meaes: " + str(meaes))

    save_model_to_file(best_model, model_path_out, lowest_err, optimizer)
    return dataset.nonzeros


def NNteach_from_python(training_data, data_path, init_weights, mission, config):
    global conf
    t.manual_seed(42)

    global device
    conf = config
    device = conf["device"]
    global batch_aug
    batch_aug = batch_augmentations.to(device)
    print("trianing:" + str(mission.name))
    image_count = teach_stuff(train_data=training_data, model_path_init=init_weights,
                              model_path_out=mission.c_strategy.model_path, out=mission.plot_folder,
                              data_path=data_path)
    mission.c_strategy.used_teach_count = image_count

    return image_count



if __name__ == '__main__':
    pass
    # teach_stuff()
