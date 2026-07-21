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

def acronym_list_gen(dict_dir):

    Allen_info_dict = np.load(dict_dir + '/acronym_Allen_info.npy', allow_pickle=True)
    Allen_info_dict = Allen_info_dict.item()
    acronym_list_Allen = [acronym for acronym in Allen_info_dict]

    Beryl_info_dict = np.load(dict_dir + '/acronym_Beryl_info.npy', allow_pickle=True)
    Beryl_info_dict = Beryl_info_dict.item()
    acronym_list_Beryl = [acronym for acronym in Beryl_info_dict]

    acronym_list = acronym_list_Allen + acronym_list_Beryl

    return acronym_list

def subject_od_ind_gen(list_dict, acronym_list, subject_num):

    '''
    This function try to find subjects which only belongs to brain regions which have subject recording larger than subject_num.
    The subject_pre contains all subjects whose brain regions are small or equal to 15.
    The subject_od_list contains all subjects which larger than 15 by using setdiff.
    '''

    subject_pre = []
    for acronym_ii, acronym in enumerate(acronym_list):
        print(acronym)
        subject0 = np.unique(np.array(list_dict['subject_list'])[np.argwhere(list_dict['brain_region_index'] == acronym_ii).flatten()])
        if len(subject0) <= subject_num:
            subject_pre.append(subject0)

    subject_od_list = np.setdiff1d(np.unique(np.array(list_dict['subject_list'])), np.unique(np.concatenate(subject_pre, axis=0)))

    subject_od_ind = []
    for subject in subject_od_list:
        subject_od_ind.append(np.argwhere(np.array(list_dict['subject_list']) == subject).flatten())
    subject_od_ind = np.concatenate(subject_od_ind, axis=0)

    return subject_od_ind, subject_od_list

def dat_ind_gen(list_dict, subject_od_ind, key):

    train_ind = np.setdiff1d(np.intersect1d(np.argwhere(np.array(list_dict['train_list_intest']) == 1).flatten(),
                                np.argwhere(np.array(list_dict['key_list']) == key).flatten()), subject_od_ind)

    valid_ind = np.setdiff1d(np.intersect1d(np.argwhere(np.array(list_dict['valid_list_intest']) == 1).flatten(),
                                np.argwhere(np.array(list_dict['key_list']) == key).flatten()), subject_od_ind)

    test_ind = np.setdiff1d(np.intersect1d(np.argwhere(np.array(list_dict['test_list_intest']) == 1).flatten(),
                                np.argwhere(np.array(list_dict['key_list']) == key).flatten()), subject_od_ind)

    test_subject_ind = np.intersect1d(subject_od_ind, np.argwhere(np.array(list_dict['key_list']) == key).flatten())

    return train_ind, valid_ind, test_ind, test_subject_ind
















