import random
import torch
import torch as t
from torch.nn.functional import conv2d, conv1d
import torch.nn as nn
import math
from einops import rearrange
from torch.nn import functional as F
from copy import deepcopy
import os
import errno
# non of libraries in this module are local. This module is independet of other scripts

def create_conv_block(in_channel, out_channel, kernel, stride, padding, pooling, bn=True, relu=True, pool_layer=False):
    net_list = [t.nn.Conv2d(in_channel, out_channel, kernel, stride, padding, padding_mode="circular")]
    if bn:
        net_list.append(t.nn.BatchNorm2d(out_channel))
    if relu:
        net_list.append(t.nn.ReLU())
    if pooling[0] > 0 or pooling[1] > 0:
        if not pool_layer:
            net_list.append(t.nn.MaxPool2d(pooling, pooling))
        else:
            net_list.append(t.nn.Conv2d(out_channel, out_channel, pooling, pooling))
    return t.nn.Sequential(*net_list)


class CNNOLD(t.nn.Module):

    def __init__(self, lp, fs, ech):
        super(CNNOLD, self).__init__()
        pad = fs // 2
        self.l1 = create_conv_block(3, 16, fs, 1, pad, (2, 2), pool_layer=lp)
        self.l2 = create_conv_block(16, 64, fs, 1, pad, (2, 2), pool_layer=lp)
        self.l3 = create_conv_block(64, 256, fs, 1, pad, (2, 2), pool_layer=lp)
        self.l4 = create_conv_block(256, 512, fs, 1, pad, (2, 1), pool_layer=lp)
        self.l5 = create_conv_block(512, ech, fs, 1, pad, (3, 1), bn=False, relu=False, pool_layer=lp)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        return x


def get_custom_CNN(lp, fs, ech):
    return CNNOLD(lp, fs, ech)


class Siamese(t.nn.Module):

    def __init__(self, backbone, padding=3, eb=False):
        super(Siamese, self).__init__()
        self.backbone = backbone
        self.padding = padding
        # self.wb = t.nn.Parameter(t.tensor([1.0, 0.0]), requires_grad=True)
        self.out_batchnorm = t.nn.BatchNorm2d(1)
        self.end_bn = eb

    def forward(self, source, target, padding=None, displac=None):
        source = self.backbone(source)
        target = self.backbone(target)
        if displac is None:
            # regular walk through
            score_map = self.match_corr(target, source, padding=padding)
            if not self.end_bn:
                score_map = (score_map - score_map.mean(dim=-1, keepdim=True)) / score_map.std(dim=-1, keepdim=True)
                score_map = score_map * self.wb[0] + self.wb[1]
            else:
                score_map = self.out_batchnorm(score_map)
            return score_map.squeeze(1).squeeze(1)
        else:
            # for importance visualisation
            shifted_target = t.roll(target, displac, -1)
            score = source * shifted_target
            score = t.sum(score, dim=[1, 2])
            return score

    def match_corr(self, embed_ref, embed_srch, padding=None):
        """ Matches the two embeddings using the correlation layer. As per usual
        it expects input tensors of the form [B, C, H, W].
        Args:
            embed_ref: (torch.Tensor) The embedding of the reference image, or
                the template of reference (the average of many embeddings for
                example).
            embed_srch: (torch.Tensor) The embedding of the search image.
        Returns:
            match_map: (torch.Tensor) The correlation between
        """

        if padding is None:
            padding = self.padding
        b, c, h, w = embed_srch.shape
        _, _, h_ref, w_ref = embed_ref.shape
        # Here the correlation layer is implemented using a trick with the
        # conv2d function using groups in order to do the correlation with
        # batch dimension. Basically we concatenate each element of the batch
        # in the channel dimension for the search image (making it
        # [1 x (B.C) x H' x W']) and setting the number of groups to the size of
        # the batch. This grouped convolution/correlation is equivalent to a
        # correlation between the two images, though it is not obvious.

        if self.training:
            match_map = conv2d(embed_srch.view(1, b * c, h, w), embed_ref, groups=b, padding=(0, padding))
            match_map = match_map.permute(1, 0, 2, 3)
        else:
            match_map = F.conv2d(F.pad(embed_srch.view(1, b * c, h, w), pad=(padding, padding, 1, 1), mode='circular'),
                                 embed_ref, groups=b)

            match_map = t.max(match_map.permute(1, 0, 2, 3), dim=2)[0].unsqueeze(2)
        return match_map


def save_model(model, optimizer,omodel,epoch):
    t.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None
    }, omodel)
    print("Model saved to: "+omodel)


def load_model(model, path, optimizer=None):
    checkpoint = t.load(path, map_location=t.device("cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded model at", path)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer
    else:
        return model


def jit_save(model, name, epoch, arb_in, args,comment=""):
    # save model arguments
    filename = "./results_" + name + "/model.info"
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, "w") as f:
        f.write(str(args))

    # save actual model
    t.jit.save(t.jit.trace(model, arb_in), "./results_" + name + "/model_" + str(epoch) +comment+ ".jit")


def jit_load(path, device):
    return t.jit.load(path, map_location=device)

