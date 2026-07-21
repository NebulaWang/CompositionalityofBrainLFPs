from natsort import natsorted
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy as sp
import scipy.signal as signal
import torchaudio
import math

import torchvision
import torchvision.transforms as transforms

import torchaudio.models as audio_models

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

def spectro_norm(brain_signal_spectro, spectro_args):

    if spectro_args['Log'] == True:
        brain_signal_spectro = torch.log(brain_signal_spectro + 1e-12)

    [N, C, H, W] = brain_signal_spectro.size()
    for ii in range(0, N):
        brain_signal_spectro[ii,:,:,:] = brain_signal_spectro[ii,:,:,:] / (
            brain_signal_spectro[ii,:,:,:].max() - brain_signal_spectro[ii,:,:,:].min() + 1e-12)

    brain_signal_spectro = 255 * brain_signal_spectro

    return brain_signal_spectro

def lfp_spectro(brain_signal_lfp, spectro_args, train_args):

    norm, temp, power = train_args['norm'], train_args['temp'], spectro_args['power']

    nfft = spectro_args['nfft']
    Resize = torchvision.transforms.Resize((1,spectro_args['sampling_lfp']))
    brain_signal_lfp = Resize(brain_signal_lfp.unsqueeze(1)).squeeze(1)

    transform = torchaudio.transforms.Spectrogram(nfft, power=power)
    brain_signal_lfp_spectro = transform(brain_signal_lfp)
    lfp_bin = spectro_args['sampling_lfp'] / 2 / (nfft // 2 + 1)
    lfp_bound = [int(spectro_args['LFP_bound'][0] // lfp_bin), int(spectro_args['LFP_bound'][1] // lfp_bin)]
    Resize = torchvision.transforms.Resize((spectro_args['LFP_img'][0], spectro_args['LFP_img'][1]))
    brain_signal_lfp_spectro = Resize(brain_signal_lfp_spectro[:,lfp_bound[0]:lfp_bound[1],:].unsqueeze(1))

    return brain_signal_lfp_spectro

