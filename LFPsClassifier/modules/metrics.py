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

# accu = match number / sample size
def accu_fun(py, y):
    return (torch.sum(torch.argmax(py, dim=1) == y) / y.size(0)).detach().cpu()
