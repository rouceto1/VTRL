#!/usr/bin/env python3.9
from .train_eval_common import *
from tqdm import tqdm
from scipy import interpolate
from torch.utils.data import DataLoader
model = None
import torch as t

def get_histogram(src, tgt, padding):
    global model
    histogram = model(src, tgt, padding)  # , fourrier=True)
    std, mean = t.std_mean(histogram, dim=-1, keepdim=True)
    if any(std == 0):
        return t.softmax(histogram, dim=1)
    histogram = (histogram - mean) / std
    histogram = t.softmax(histogram, dim=1)
    return histogram


def eval_displacement(eval_model=None, model_path=None,
                      loader=None, histograms=None, conf=None, padding=None):
    """
    :param model_path:
    :param histograms:
    :param padding:
    :param eval_model: :param model_path: if model is given in eval_model model path is not used,
    :param loader:
    :return:
    """
    global model
    model, conf = get_model(model, model_path, eval_model, conf, padding)
    train_loader = loader
    device = conf["device"]
    model.eval()
    with t.no_grad():
        abs_err = 0
        valid = 0
        idx = 0
        errors = []
        results = []
        for batch in tqdm(train_loader):
            # batch: source, cropped_target, heatmap, data_idx, original_image, displ
            if len(batch) > 3:
                output_size = conf["output_size"]
                source, target, gt = (batch[0].to(device)), (batch[1].to(device)), batch[5]
                if abs(gt.numpy()[0]) >= conf["width"]:
                    print("should not happen")
                    continue
            else:
                output_size = conf["output_size"] - 1
                source, target = (batch[0].to(device)), (batch[2].to(device))
                gt = 0
            histogram = get_histogram(source, target, padding)
            shift_hist = histogram.cpu()
            tmp_idx = 0
            if histograms is not None:  # only run this when in pure eval
                for hist in shift_hist:
                    histograms[idx+tmp_idx, :] = hist.cpu().numpy()
                    tmp_idx += 1
            else:
                for hist in shift_hist:
                    tmp_idx += 1
            f = interpolate.interp1d(np.linspace(0, conf["width"], output_size), shift_hist, kind="cubic")
            interpolated = f(np.arange(conf["width"]))
            ret = -(np.argmax(interpolated) - conf["width"] / 2) / conf["width"]
            results.append(ret)
            displac_mult = 1024 / conf["width"]  # TODO wtf is this magic
            # tmp_err = (ret - gt.numpy()[0]) / displac_mult
            tmp_err = (ret - gt) / displac_mult
            abs_err += abs(tmp_err)
            errors.append(tmp_err)
            # if abs(ret - gt.numpy()[0]) < conf["tolerance"]:
            if abs(ret - gt) < conf["tolerance"]:
                valid += 1
            idx += tmp_idx


            if idx > conf["eval_limit"]:
                break

        return abs_err / idx, valid * 100 / idx, histograms, results


def NNeval_from_python(files, data_path, weights_file, config=None):
    print("evaluating:" + str(weights_file))
    global conf
    global device
    conf = config
    device = conf["device"]
    dataset, histograms = get_dataset(data_path, files, conf)
    loader = DataLoader(dataset, conf["batch_size_eval"], shuffle=False)
    return eval_displacement(model_path=weights_file, loader=loader,
                             histograms=histograms, conf=conf, padding=conf["pad_eval"])


if __name__ == '__main__':
    eval_displacement()
