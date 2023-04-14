#!/usr/bin/env python3.9
from .train_eval_common import *

conf = load_config("./NN_config.yaml", 512)
device = conf["device"]
# MODEL = "model_47"
res = int(conf["image_height"] * conf["size_frac"])
#transform = Resize(res)
# transform = Resize(192)
# crops_num = int((conf["width"] // conf["crop_size"]) * conf["crops_multiplier"])
# crops_idx = np.linspace(0, conf["width"] - conf["crop_size"], crops_num, dtype=int) + conf["fraction"] // 2
model = None


def get_histogram(src, tgt, padding):
    global model
    ##maybe global model needed?? TODO
    #tgt = tgt[...,4:conf["width"]-4]
    histogram = model(src, tgt, padding)  # , fourrier=True)
    std, mean = t.std_mean(histogram, dim=-1, keepdim=True)
    if std == 0:
        return t.softmax(histogram, dim=1)
    histogram = (histogram - mean) / std
    histogram = t.softmax(histogram, dim=1)
    return histogram


def eval_displacement(eval_model=None, model_path=None,
                      data_path="strand", loader=None, histograms=None, padding=None):
    '''
    :param eval_model: :param model_path: if model is given in eval_model model path is not used,
    :param data_path:
    :param GT:
    :param loader:
    :return:
    '''
    global conf
    global model
    model, conf = get_model(model, model_path, eval_model, conf, padding)
    train_loader = loader
    model.eval()
    with torch.no_grad():
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
            if not histograms is None: # only run this when in pure eval
                histograms[idx, :] = shift_hist.cpu().numpy()
            f = interpolate.interp1d(np.linspace(0, conf["width"], output_size), shift_hist, kind="cubic")
            interpolated = f(np.arange(conf["width"]))
            ret = -(np.argmax(interpolated) - conf["width"] / 2)/conf["width"]
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
                             histograms=histograms, padding=conf["pad_eval"])


if __name__ == '__main__':
    eval_displacement()
