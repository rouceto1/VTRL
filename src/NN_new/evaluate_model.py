#!/usr/bin/env python3.9
from .train_eval_common import *
from tqdm import tqdm
from scipy import interpolate
from torch.utils.data import DataLoader

model = None
import torch as t
from .utils import plot_histogram


def get_histogram(src, tgt, padding, model):
    histogram = model(src, tgt, padding)  # , fourrier=True)
    std, mean = t.std_mean(histogram, dim=-1, keepdim=True)
    if any(std == 0):
        return t.softmax(histogram, dim=1)
    histogram = (histogram - mean) / std
    histogram = t.softmax(histogram, dim=1)
    return histogram


def eval_displacement(eval_model=None, model_path=None,
                      loader=None, histograms=None, conf=None, batch_size=1,
                      padding=None, epoch=None,
                      plot_path=None, plot_name=None, is_teaching = False):
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
    count = 0
    with t.no_grad():
        abs_err = 0
        valid = 0
        idx = 0
        errors = []
        results = []
        for batch in tqdm(train_loader):
            # batch: source, cropped_target, heatmap, data_idx, original_image, displ
            if is_teaching:
                output_size = conf["output_size"]
                source, target, gt = (batch[0].to(device)), (batch[1].to(device)), batch[5]
            else:
                output_size = conf["output_size"] - 1
                source, target = (batch[0].to(device)), (batch[3].to(device))
                gt = [0] * batch_size

            histogram = get_histogram(source, target, padding, model)
            shift_hist = histogram.cpu()
            tmp_idx = 0
            if histograms is not None:  # only run this when in pure eval
                for hist in shift_hist:
                    histograms[idx + tmp_idx, :] = hist.cpu().numpy()
                    tmp_idx += 1
            else:
                for hist in shift_hist:
                    tmp_idx += 1
            for i, h in enumerate(shift_hist):
                if type(gt) == int:
                    gt_list = [gt]
                else:
                    gt_list = gt
                f = interpolate.interp1d(np.linspace(0, conf["width"], output_size), h, kind="cubic")
                interpolated = f(np.arange(conf["width"]))
                ret = -(np.argmax(interpolated) - conf["width"] / 2) / conf["width"]
                results.append(ret)
                tmp_err = (ret - gt_list[i])
                abs_err += abs(tmp_err)
                errors.append(tmp_err)

                # if abs(ret - gt.numpy()[0]) < conf["tolerance"]:
                if abs(ret - gt_list[i]) < conf["tolerance"]:
                    valid += 1
                idx += 1
            if idx > conf["eval_limit"]:
                break
            if conf["plot_eval_in_train"] or conf["plot_eval"]:
                plot_source = source[0].cpu()
                plot_target = target[0].cpu()
                plot_hist = shift_hist[0]
                croped = None
                if is_teaching:
                    croped = target[0].cpu()
                if idx < 5:
                    plot_histogram(plot_source, plot_target, displacement=ret, cropped_target=croped, histogram=plot_hist,
                                   name=plot_name + "_" + str(epoch) + "_" + str(idx),
                                   dir=plot_path)

        return abs_err / idx, valid * 100 / idx, histograms, results


def NNeval_from_python(files, data_path, out_path, weights_file, config=None):
    print("evaluating:" + str(weights_file))
    global conf
    global device
    conf = config
    device = conf["device"]
    dataset, histograms = get_dataset(data_path, files, conf)
    loader = DataLoader(dataset, conf["batch_size_eval"], shuffle=False, pin_memory=True, num_workers=10)
    mae, acc, hist, disp = eval_displacement(model_path=weights_file, loader=loader,
                                             histograms=histograms, conf=conf, batch_size=conf["batch_size_eval"],
                                             padding=conf["pad_eval"],
                                             plot_path=out_path,
                                             plot_name="eval_hist", epoch="max"
                                             )
    return mae, acc, hist, disp


if __name__ == '__main__':
    eval_displacement()
