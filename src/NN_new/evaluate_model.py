#!/usr/bin/env python3.9
from .train_eval_common import *


conf = load_config("/home/rouceto1/git/VTRL/NN_config.yaml", 512)
device = conf["device"]
# MODEL = "model_47"
res = int(conf["image_height"] * conf["size_frac"])
transform = Resize(res)
# transform = Resize(192)
# crops_num = int((conf["width"] // conf["crop_size"]) * conf["crops_multiplier"])
# crops_idx = np.linspace(0, conf["width"] - conf["crop_size"], crops_num, dtype=int) + conf["fraction"] // 2

model = None


def get_histogram_old(src, tgt):
    target_crops = []
    for crop_idx in crops_idx:
        target_crops.append(tgt[..., crop_idx:crop_idx + conf["crop_size"]])
    target_crops = t.cat(target_crops, dim=0)
    batched_source = src.repeat(crops_num // conf["batching"], 1, 1, 1)
    histogram = model(batched_source, target_crops, padding=conf["pad"])  # , fourrier=True)
    histogram = t.softmax(histogram, dim=1)
    return histogram


def get_histogram(src, tgt, padding):
    histogram = model(src, tgt, padding)  # , fourrier=True)
    std, mean = t.std_mean(histogram, dim=-1, keepdim=True)
    histogram = (histogram - mean) / std
    histogram = t.softmax(histogram, dim=1)
    return histogram


def eval_displacement(eval_model=None, model_path=None,
                      data_path="strand", loader=None, histograms=[], hist_padding=None):
    '''
    :param eval_model: :param model_path: if model is given in eval_model model path is not used,
    :param data_path:
    :param GT:
    :param loader:
    :return:
    '''
    global conf
    global model
    model, conf = get_model(model, model_path, eval_model, conf)

    train_loader = loader
    model.eval()
    with torch.no_grad():
        abs_err = 0
        valid = 0
        idx = 0
        errors = []
        results = []
        for batch in tqdm(train_loader):
            # bathch: source, cropped_target, heatmap, data_idx, original_image, displ
            if len(batch) > 2:
                source, target, gt = transform(batch[0].to(device)), transform(batch[4].to(device)), batch[5]
                if abs(gt.numpy()[0]) > 256:
                    print("should not happen")
                    continue
            else:
                source, target = transform(batch[0].to(device)), transform(batch[1].to(device))
                gt = 0
            histogram = get_histogram(source, target, hist_padding)
            shift_hist = histogram.cpu()
            if loader is None:
                histograms[idx, :] = shift_hist.cpu().numpy()
            f = interpolate.interp1d(np.linspace(0, conf["width"], conf["output_size"] - 1), shift_hist, kind="cubic")
            interpolated = f(np.arange(conf["width"]))
            ret = -(np.argmax(interpolated) - conf["width"] / 2)
            results.append(ret)
            displac_mult = 1024 / conf["width"]  ###TODO wtf is this magic
            # tmp_err = (ret - gt.numpy()[0]) / displac_mult
            tmp_err = (ret - gt) / displac_mult
            abs_err += abs(tmp_err)
            errors.append(tmp_err)
            # if abs(ret - gt.numpy()[0]) < conf["tolerance"]:
            if abs(ret - gt) < conf["tolerance"]:
                valid += 1
            idx += 1

            if idx > conf["eval_limit"]:
                break

        return abs_err / idx, valid * 100 / idx, histograms, results


def NNeval_from_python(files, data_path, weights_file):
    print("evaluating:" + str(weights_file))
    global conf
    dataset, histograms = get_dataset(data_path, files, conf)
    loader = DataLoader(dataset, 1, shuffle=False)
    return eval_displacement(data_path=data_path, model_path=weights_file, loader=loader,
                             histograms=histograms, hist_padding=conf["histpad_eval"])


if __name__ == '__main__':
    eval_displacement()
