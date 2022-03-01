import torch
import torch as t
from model import Siamese, load_model, get_custom_CNN, jit_load
from torch.utils.data import DataLoader
from parser_grief import ImgPairDataset # path var is set for this module
# from parser_grief import CroppedImgPairDataset # this module is not used in this code
from torchvision.transforms import Resize
from tqdm import tqdm
from utils import get_shift, plot_samples, plot_displacement, affine
import numpy as np
from scipy import interpolate


device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
# device = t.device("cpu")
#path="/home/users/aamjadi/hdd/IROS_data_set/grief_jpg"
path="/home/arash/Desktop/workdir/IROS/grief_jpg"
data_path=path
DATASET ="stromovka"
VISUALIZE = False
PLOT_IMPORTANCES = False
WIDTH = 512 #! nord width is 512 but stromovka width is 1024
CROP_SIZE = WIDTH - 8
PAD = (CROP_SIZE - 8) // 16
FRACTION = 8
OUTPUT_SIZE = 64  #  WIDTH // FRACTION
CROPS_MULTIPLIER = 1
BATCHING = CROPS_MULTIPLIER    # this improves evaluation speed by a lot
MASK = t.zeros(OUTPUT_SIZE)
MASK[:PAD] = t.flip(t.arange(0, PAD), dims=[0])
MASK[-PAD:] = t.arange(0, PAD)
MASK = OUTPUT_SIZE - 1 - MASK
MASK = (OUTPUT_SIZE - 1) / MASK.to(device)
LAYER_POOL = True
# FILTER_SIZE = 3 #! incompatible with train_siam.py
FILTER_SIZE =5 
END_BN = True
EMB_CHANNELS = 256

EVAL_LIMIT = 2000
TOLERANCE = 48

# MODEL_TYPE = "siam_cnn"
MODEL_TYPE = "siam"
# MODEL = "model_150_noBN"
# MODEL = "model_47"
MODEL = "model_24.pt" #! extention better be written in model name
model = None

transform = Resize(192)
# transform = Resize(192 * 2)
# transform = Resize((288, 512))
crops_num = int((WIDTH // CROP_SIZE) * CROPS_MULTIPLIER)
crops_idx = np.linspace(0, WIDTH-CROP_SIZE, crops_num, dtype=int) + FRACTION // 2

# crops_idx = np.array([WIDTH // 2 - CROP_SIZE // 2])
# crops_num = 1
print(crops_num, np.array(crops_idx))


histograms = np.zeros((1000, 64))


def get_histogram(src, tgt):
    target_crops = []
    for crop_idx in crops_idx:
        target_crops.append(tgt[..., crop_idx:crop_idx + CROP_SIZE])
    target_crops = t.cat(target_crops, dim=0)
    batched_source = src.repeat(crops_num // BATCHING, 1, 1, 1)
    # batched_source = t.zeros_like(batched_source)
    # batched_source = src
    histogram = model(batched_source, target_crops, padding=PAD)  # , fourrier=True)
    # histogram = histogram * MASK
    # histogram = t.sigmoid(histogram)
    # std, mean = t.std_mean(histogram, dim=-1, keepdim=True)
    # histogram = (histogram - mean) / std
    histogram = t.softmax(histogram, dim=1)
    return histogram


def get_importance(src, tgt, displac):
    # displac here is in size of embedding (OUTPUT_SIZE)
    histogram = model(src, tgt, padding=PAD, displac=displac).cpu().numpy()
    f = interpolate.interp1d(np.linspace(0, 512, OUTPUT_SIZE), histogram, kind="cubic")
    interpolated = f(np.arange(512))
    return interpolated[0]


def eval_displacement(eval_model=None, path="./results_" + MODEL_TYPE + "/" + MODEL ):
    global model
    # backbone = get_custom_CNN(LAYER_POOL, FILTER_SIZE, EMB_CHANNELS)
    # model = Siamese(backbone, padding=PAD, eb=END_BN).to(device)
    # model = load_model(model, "./results_" + MODEL_TYPE + "/" + MODEL)
    
    if eval_model is not None:
        model = eval_model
    else:
        # model = jit_load(path, device) # forward() expected at most 3 argument(s) but received 4 argument(s)
        
        # model=load_model(model, path)
        # some_data=model[1]
        # model=model[0]

        backbone = get_custom_CNN(LAYER_POOL, FILTER_SIZE, EMB_CHANNELS)
        model = Siamese(backbone, padding=PAD, eb=END_BN).to(device)
        model = load_model(model, "./results_" + MODEL_TYPE + "/" + MODEL)
    
    dataset = ImgPairDataset(path=data_path,dataset=DATASET)
    train_loader = DataLoader(dataset, 1, shuffle=False)

    model.eval()
    with torch.no_grad():
        abs_err = 0
        valid = 0
        idx = 0
        errors = []
        results = []
        for batch in tqdm(train_loader):
            source, target, displ = transform(batch[0].to(device)), transform(batch[1].to(device)), batch[2]
            #! imshow(batch[1][0].numpy().T) # for curiosity
            # do it in both directions target -> source and source -> target
            histogram = get_histogram(source, target)
            # shift_hist = get_shift(WIDTH, CROP_SIZE, histogram, crops_idx)
            # histograms[idx, :] = histogram.cpu().numpy()
            shift_hist = histogram.cpu()
            # histogram = get_histogram(target, source)
            # shift_hist += t.flip(get_shift(WIDTH, CROP_SIZE, histogram, crops_idx), dims=(-1, ))
            f = interpolate.interp1d(np.linspace(0, 1024, OUTPUT_SIZE), shift_hist, kind="cubic") #? why this is needed?
            interpolated = f(np.arange(1024))
            # interpolated = np.interp(np.arange(0, 1024), np.linspace(0, 1024, OUTPUT_SIZE), shift_hist.numpy())
            ret = -(np.argmax(interpolated) - 512)
            results.append(ret)
            displac_mult = 1024/WIDTH #? what is this?
            tmp_err = (ret - displ.numpy()[0])/displac_mult
            abs_err += abs(tmp_err)
            errors.append(tmp_err)
            if VISUALIZE and abs(ret - displ.numpy()[0]) >= TOLERANCE:
                if PLOT_IMPORTANCES:
                    importances = get_importance(source, target, int(ret/(FRACTION*displac_mult)))
                else:
                    importances = None
                plot_displacement(source.squeeze(0).cpu(),
                                  target.squeeze(0).cpu(),
                                  shift_hist.squeeze(0).cpu(),
                                  displacement=-displ.numpy()[0]/displac_mult,
                                  importance=importances,
                                  name=str(idx),
                                  dir="results_" + MODEL_TYPE + "/eval_" + MODEL + "/")
                print(ret, displ.numpy()[0])

            idx += 1
            if abs(ret - displ.numpy()[0]) < TOLERANCE:
                valid += 1

            if idx > EVAL_LIMIT:
                break

        print("Evaluated:", "\nAbsolute mean error:", abs_err/idx, "\nPredictions in tolerance:", valid*100/idx, "%")
        # np.savetxt("results_" + MODEL_TYPE + "/eval_" + MODEL + "/" + DATASET + "_errors.csv", np.array(errors) * 2.0, delimiter=",")
        # np.savetxt("results_" + MODEL_TYPE + "/eval_" + MODEL + "/" + DATASET + "_preds.csv", np.array(errors) * 2.0, delimiter=",")
        # np.savetxt("results_" + MODEL_TYPE + "/eval_" + MODEL + "/" + DATASET + "_histograms.csv", histograms, delimiter=",")
        return abs_err/idx, valid*100/idx


if __name__ == '__main__':
    eval_displacement()
