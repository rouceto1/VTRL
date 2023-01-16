import torch as t
from .model import Siamese, load_model, get_custom_CNN, save_model
from torch.utils.data import DataLoader
from .parser_nordland import RectifiedNordland
from .parser_strands import Strands
from tqdm import tqdm
import numpy as np
from scipy import interpolate
from torch.optim import SGD, AdamW
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
import argparse

parser = argparse.ArgumentParser(
    description='EXAMPLE: --in_model_path ./results_siam/model.pt --data_path ../nordland_rectified/ --d0 fall --d1 winter --dsp ../nordland_rectified/orb_orb_su_w_PRE_FILTER_ENTR.csv --omodel ./results_siam/model1.pt')
parser.add_argument('--in_model_path', type=str,
                    help="path to the model to be train. if passed string start , it will start with an empty model")
parser.add_argument('--data_path', type=str, help="path to the main data")
parser.add_argument('--d0', type=str, help="path to first dataset")
parser.add_argument('--d1', type=str, help="path to second dataset")
parser.add_argument('--dsp', type=str, help="path to displacement file")
parser.add_argument('--omodel', type=str, help="path to output model")
args = parser.parse_args()
in_model_path = args.in_model_path
d0 = args.d0
d1 = args.d1
dsp = args.dsp
omodel = args.omodel
data_path = args.data_path
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
# print("[+] device is {}".format(device))
LR = 2e-5
EPOCHS = 3
EVAL_RATE = 1
BATCH_SIZE = 30
THRESHOLD = 0.25
VISUALIZE = False
PLOT_IMPORTANCES = False
WIDTH = 512  # ! nord width is 512 but stromovka width is 1024
# CROP_SIZE = WIDTH - 8
CROP_SIZE = 56
SMOOTHNESS = 3
PAD = (CROP_SIZE - 8) // 16
FRACTION = 8
OUTPUT_SIZE = 64  # WIDTH // FRACTION
CROPS_MULTIPLIER = 1
BATCHING = CROPS_MULTIPLIER  # this improves evaluation speed by a lot
LAYER_POOL = False
FILTER_SIZE = 5
END_BN = True
EMB_CHANNELS = 256
EVAL_LIMIT = 2000
TOLERANCE = 48


def get_histogram(src, tgt):
    global model
    # ! if for loop below runs, it will try to crop the target from different parts. definitly sth that we dont want
    # target_crops = []
    # for crop_idx in crops_idx:
    #     target_crops.append(tgt[..., crop_idx:crop_idx + CROP_SIZE])
    # target_crops = t.cat(target_crops, dim=0)
    # batched_source = src.repeat(crops_num // BATCHING, 1, 1, 1)
    # histogram = model(batched_source, target_crops, padding=PAD)  # , fourrier=True)
    # histogram = model(batched_source, tgt, padding=PAD)  # , fourrier=True)
    histogram = model(src, tgt, padding=PAD)  # , fourrier=True)
    histogram = t.softmax(histogram, dim=1)
    return histogram


def get_importance(src, tgt, displac):
    # displac here is in size of embedding (OUTPUT_SIZE)
    histogram = model(src, tgt, padding=PAD, displac=displac).cpu().numpy()
    f = interpolate.interp1d(np.linspace(0, 512, OUTPUT_SIZE), histogram, kind="cubic")
    interpolated = f(np.arange(512))
    return interpolated[0]


def eval_displacement(eval_model):
    global model
    model = eval_model
    dataset = RectifiedNordland(CROP_SIZE, FRACTION, SMOOTHNESS, data_path, dsp, [d0, d1], threshold=THRESHOLD)  # !
    train_loader = DataLoader(dataset, 1, shuffle=False)
    model.eval()
    with t.no_grad():
        abs_err = 0
        valid = 0
        idx = 0
        errors = []
        results = []
        results_gt = []
        with open(dsp, 'r') as f:
            hist_height = len(f.read().split("\n"))
        histograms = np.zeros((hist_height, 64))
        # ! hist rows with zero rows means that they are discarded from evaluation
        for batch in train_loader:
            # source, target, displ = transform(batch[0].to(device)), transform(batch[1].to(device)), batch[2] #! transform changes the shape
            source, target, displ, data_idx = batch[0].to(device), batch[1].to(device), batch[2], batch[
                3]  # ! without transform hist wont be 64
            histogram = get_histogram(source, target)
            histograms[data_idx, :] = histogram.cpu().numpy()
            shift_hist = histogram.cpu()
            f = interpolate.interp1d(np.linspace(0, 1024, OUTPUT_SIZE), shift_hist,
                                     kind="cubic")  # ? why this is needed?
            interpolated = f(np.arange(1024))
            ret = -(np.argmax(interpolated) - 512)
            results.append(ret)
            displac_mult = 1024 / WIDTH  # ? what is this?

            f_gt = interpolate.interp1d(np.linspace(0, 1024, OUTPUT_SIZE), displ, kind="cubic")  # ? why this is needed?
            interpolated_gt = f_gt(np.arange(1024))
            ret_gt = -(np.argmax(interpolated_gt) - 512)
            results_gt.append(ret_gt)

            tmp_err = (ret - ret_gt) / displac_mult
            abs_err += abs(tmp_err)
            errors.append(tmp_err)
            if VISUALIZE and abs(ret - ret_gt) >= TOLERANCE:
                if PLOT_IMPORTANCES:
                    importances = get_importance(source, target, int(ret / (FRACTION * displac_mult)))
                else:
                    importances = None
                # plot_displacement(source.squeeze(0).cpu(),
                #                   target.squeeze(0).cpu(),
                #                   shift_hist.squeeze(0).cpu(),
                #                   displacement=ret_gt,
                #                   importance=importances,
                #                   name=str(idx),
                #                   dir="results_" + MODEL_TYPE + "/eval_" + MODEL + "/")
                # print(ret, ret_gt)

            idx += 1
            if abs(ret - ret_gt) < TOLERANCE:
                valid += 1
            # ! I dont know why zdenec wrote here
            # if idx > EVAL_LIMIT:
            # break

        print("Evaluated: Absolute mean error: {} Predictions in tolerance: {} %".format(abs_err / idx,
                                                                                         valid * 100 / idx))
        return abs_err / idx, valid * 100 / idx


def get_dataset(data_path, GT):
    if "nordland" in data_path:
        dataset = RectifiedNordland(CROP_SIZE, FRACTION, SMOOTHNESS, data_path, dsp, [d0, d1], threshold=THRESHOLD)
    elif "stromovka" in data_path:
        pass
    elif "carlevaris" in data_path:
        pass
    elif "strand" in data_path:

        dataset = Strands(CROP_SIZE, FRACTION, SMOOTHNESS, GT, thre=THRESHOLD)
    return dataset


def train_loop(epoch, GT=0, data_path="", ):
    global PAD, model, optimizer, loss
    NEGATIVE_FRAC = 1 / 3
    model.train()
    loss_sum = 0
    generation = 0
    # dataset = RectifiedNordland(CROP_SIZE,FRACTION, SMOOTHNESS ,data_path,dsp,[d0,d1],threshold=THRESHOLD) #!
    dataset = get_dataset(data_path, GT)
    train_loader = DataLoader(dataset, BATCH_SIZE, shuffle=False)
    idx = 0
    try:
        for batch in tqdm(train_loader):
            source, target, heatmap = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            # ! plt.imshow(batch[0][0].numpy().T) # for curiosity
            # source = batch_augmentations(source) #! no augmentation with gt
            # ! if NEGATIVE_FRAC > 0.01:
            # ! batch, heatmap = hard_negatives(source, heatmap)
            out = model(source, target, padding=PAD)
            # print(source, target)
            optimizer.zero_grad()
            l = loss(out, heatmap)
            loss_sum += l.cpu().detach().numpy()
            l.backward()
            optimizer.step()
            idx += 1
            # if VISUALIZE:
            # plot_samples(source[0].cpu(),
            #         target[0].cpu(),
            #         heatmap[0].cpu(),
            #         prediction=out[0].cpu(),
            #         name=str(idx),
            #         dir="results_" + NAME + "/" + str(epoch) + "/")
    except Exception as e:
        print(e)

        print(GT[idx + 1])

    print("Training of epoch", epoch, "ended with loss", loss_sum / len(train_loader))


def NNteach_from_python(GT, data_path, weights_file, epochs):
    global backbone, model, optimizer, loss
    backbone = get_custom_CNN(LAYER_POOL, FILTER_SIZE, EMB_CHANNELS)
    model = Siamese(backbone, padding=PAD, eb=END_BN).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    loss = BCEWithLogitsLoss()
    for epoch in range(EPOCHS):
        train_loop(epoch, GT, data_path)
        save_model(model, optimizer, weights_file, epoch)  # !


if __name__ == '__main__':
    backbone = get_custom_CNN(LAYER_POOL, FILTER_SIZE, EMB_CHANNELS)
    model = Siamese(backbone, padding=PAD, eb=END_BN).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    loss = BCEWithLogitsLoss()
    if in_model_path != "start":
        load_model(model, in_model_path, optimizer)
    train_loop(0, dsp, data_path)
    save_model(model, optimizer, omodel, 0)  # !
