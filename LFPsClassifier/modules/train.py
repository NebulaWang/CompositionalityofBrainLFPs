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

import time

lib_dir = '/content/drive/MyDrive/Project/BrainRegionId/Project44/Code'
os.chdir(lib_dir)
print('Library directory: '+ lib_dir)
from modules.signal import spectro_norm, lfp_spectro
from modules.metrics import accu_fun

def net_train_AnyNet_L(train_iter, valid_iter, Classifier, spectro_args, train_args, key, ind, device):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Classifier.parameters(), train_args['lr'])
    # print('lr: ' + train_args['lr'])
    acu_array_train = []
    acu_array_valid = []
    time_array_train = []
    time_array_valid = []
    acu_valid_max = 0
    for epoch in range(0, train_args['epochs']):
        Classifier.train()
        acu_train = []
        epoch0 = 0
        time0_train = time.time()
        for x_train1, y_train, coordinate_train in train_iter:
            x_train = lfp_spectro(x_train1, spectro_args, train_args)
            y_train = y_train.to(device)
            py_train = Classifier(x_train.to(device))
            del x_train, x_train1
            if epoch0 % 800 == 0:
                if epoch0 == 0 and epoch == 0:
                    if accu_fun(py_train, y_train) == (torch.sum(torch.argmax(py_train, dim=1) == y_train) / y_train.size(0)).detach().cpu():
                        print('accu_fun is correct>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    else:
                        print('accu_fun is wrong>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                        return
                print(accu_fun(py_train, y_train))
            L = loss_fn(py_train,y_train.to(device))
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            acu_train.append(accu_fun(py_train, y_train))
            epoch0 += 1
        print(f'Acu Train: {torch.mean(torch.tensor(acu_train))}')

        acu_array_train.append(torch.mean(torch.tensor(acu_train)))
        time_array_train.append(time.time() - time0_train)

        Classifier.eval()
        acu_valid = []
        time0_valid = time.time()
        for x_valid1, y_valid, coordinate_valid in valid_iter:
            x_valid = lfp_spectro(x_valid1, spectro_args, train_args)
            y_valid = y_valid.to(device)
            py_valid = Classifier(x_valid.to(device))
            del x_valid, x_valid1
            acu_valid.append(accu_fun(py_valid, y_valid))
            # acu_valid.append((torch.sum(torch.argmax(py_valid, dim=1) == y_valid) / y_valid.size(0)).detach().cpu())

        print(f'Acu valid: {torch.mean(torch.tensor(acu_valid))}')
        if torch.mean(torch.tensor(acu_train)) > train_args['overfitting_thres']:
            if acu_valid_max < torch.mean(torch.tensor(acu_valid)):
                torch.save(Classifier, train_args['save_dir'] + f'/Model/Allen/AnyNet_L_Allen_{key}_{ind}.pth')
                torch.save(epoch, train_args['save_dir'] + f'/Model/Allen/AnyNet_L_Allen_{key}_epoch{ind}.pt')
                acu_valid_max = torch.mean(torch.tensor(acu_valid))

        acu_array_valid.append(torch.mean(torch.tensor(acu_valid)))
        time_array_valid.append(time.time() - time0_valid)

    torch.save(acu_array_train, train_args['save_dir'] + f'/Model/Allen/AnyNet_L_Allen_{key}_train_acu{ind}.pt')
    torch.save(acu_array_valid, train_args['save_dir'] + f'/Model/Allen/AnyNet_L_Allen_{key}_valid_acu{ind}.pt')

    torch.save(time_array_train, train_args['save_dir'] + f'/Model/Allen/AnyNet_L_time_array_train_{key}_{ind}.pt')
    torch.save(time_array_valid, train_args['save_dir'] + f'/Model/Allen/AnyNet_L_time_array_valid_{key}_{ind}.pt')

    return


def net_train_ViT_L(train_iter, valid_iter, Classifier, spectro_args, train_args, key, ind, device):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Classifier.parameters(), train_args['lr'])
    # print('lr: ' + train_args['lr'])
    acu_array_train = []
    acu_array_valid = []
    time_array_train = []
    time_array_valid = []
    acu_valid_max = 0
    for epoch in range(0, train_args['epochs']):
        Classifier.train()
        acu_train = []
        epoch0 = 0
        time0_train = time.time()
        for x_train1, y_train, coordinate_train in train_iter:
            x_train = lfp_spectro(x_train1, spectro_args, train_args)
            y_train = y_train.to(device)
            py_train = Classifier(x_train.to(device))
            del x_train, x_train1
            if epoch0 % 800 == 0:
                print((torch.sum(torch.argmax(py_train, dim=1) == y_train) / y_train.size(0)).detach().cpu())
            L = loss_fn(py_train,y_train.to(device))
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            acu_train.append((torch.sum(torch.argmax(py_train, dim=1) == y_train) / y_train.size(0)).detach().cpu())
            epoch0 += 1
        print(f'Acu Train: {torch.mean(torch.tensor(acu_train))}')

        acu_array_train.append(torch.mean(torch.tensor(acu_train)))
        time_array_train.append(time.time() - time0_train)

        Classifier.eval()
        acu_valid = []
        time0_valid = time.time()
        for x_valid1, y_valid, coordinate_valid in valid_iter:
            x_valid = lfp_spectro(x_valid1, spectro_args, train_args)
            y_valid = y_valid.to(device)
            py_valid = Classifier(x_valid.to(device))
            del x_valid, x_valid1
            acu_valid.append((torch.sum(torch.argmax(py_valid, dim=1) == y_valid) / y_valid.size(0)).detach().cpu())

        print(f'Acu valid: {torch.mean(torch.tensor(acu_valid))}')
        if torch.mean(torch.tensor(acu_train)) > train_args['overfitting_thres']:
            if acu_valid_max < torch.mean(torch.tensor(acu_valid)):
                torch.save(Classifier, train_args['save_dir'] + f'/Model/Allen/ViT_L_Allen_{key}_{ind}.pth')
                torch.save(epoch, train_args['save_dir'] + f'/Model/Allen/ViT_L_Allen_{key}_epoch{ind}.pt')
                acu_valid_max = torch.mean(torch.tensor(acu_valid))

        acu_array_valid.append(torch.mean(torch.tensor(acu_valid)))
        time_array_valid.append(time.time() - time0_valid)

    torch.save(torch.tensor(acu_array_train), train_args['save_dir'] + f'/Model/Allen/ViT_L_Allen_{key}_train_acu{ind}.pt')
    torch.save(torch.tensor(acu_array_valid), train_args['save_dir'] + f'/Model/Allen/ViT_L_Allen_{key}_valid_acu{ind}.pt')

    torch.save(time_array_train, train_args['save_dir'] + f'/Model/Allen/ViT_L_time_array_train_{key}_{ind}.pt')
    torch.save(time_array_valid, train_args['save_dir'] + f'/Model/Allen/ViT_L_time_array_valid_{key}_{ind}.pt')

    return

def net_train_RNN_L(train_iter, valid_iter, Classifier, spectro_args, train_args, key, ind, device):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Classifier.parameters(), train_args['lr'])
    # print('lr: ' + train_args['lr'])
    acu_array_train = []
    acu_array_valid = []
    time_array_train = []
    time_array_valid = []
    acu_valid_max = 0
    for epoch in range(0, train_args['epochs']):
        Classifier.train()
        acu_train = []
        time0_train = time.time() 
        epoch0 = 0
        for x_train1, y_train, coordinate_train in train_iter:
            x_train = lfp_spectro(x_train1, spectro_args, train_args).squeeze(1).permute(0, 2, 1)
            y_train = y_train.to(device)
            py_train = Classifier(x_train.to(device))
            del x_train, x_train1
            if epoch0 % 800 == 0:
                print((torch.sum(torch.argmax(py_train, dim=1) == y_train) / y_train.size(0)).detach().cpu())
            L = loss_fn(py_train,y_train.to(device))
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            acu_train.append((torch.sum(torch.argmax(py_train, dim=1) == y_train) / y_train.size(0)).detach().cpu())
            epoch0 += 1
        print(f'Acu Train: {torch.mean(torch.tensor(acu_train))}')

        acu_array_train.append(torch.mean(torch.tensor(acu_train)))
        time_array_train.append(time.time() - time0_train)

        Classifier.eval()
        acu_valid = []
        time0_valid = time.time()
        for x_valid1, y_valid, coordinate_valid in valid_iter:
            x_valid = lfp_spectro(x_valid1, spectro_args, train_args).squeeze(1).permute(0, 2, 1)
            y_valid = y_valid.to(device)
            py_valid = Classifier(x_valid.to(device))
            del x_valid, x_valid1
            acu_valid.append((torch.sum(torch.argmax(py_valid, dim=1) == y_valid) / y_valid.size(0)).detach().cpu())

        print(f'Acu valid: {torch.mean(torch.tensor(acu_valid))}')
        if torch.mean(torch.tensor(acu_train)) > train_args['overfitting_thres']:
            if acu_valid_max < torch.mean(torch.tensor(acu_valid)):
                torch.save(Classifier, train_args['save_dir'] + f'/Model/Allen/RNN_L_Allen_{key}_{ind}.pth')
                torch.save(epoch, train_args['save_dir'] + f'/Model/Allen/RNN_L_Allen_{key}_epoch{ind}.pt')
                acu_valid_max = torch.mean(torch.tensor(acu_valid))


        acu_array_valid.append(torch.mean(torch.tensor(acu_valid)))
        time_array_valid.append(time.time() - time0_valid)

    torch.save(torch.tensor(acu_array_train), train_args['save_dir'] + f'/Model/Allen/RNN_L_Allen_{key}_train_acu{ind}.pt')
    torch.save(torch.tensor(acu_array_valid), train_args['save_dir'] + f'/Model/Allen/RNN_L_Allen_{key}_valid_acu{ind}.pt')

    torch.save(time_array_train, train_args['save_dir'] + f'/Model/Allen/RNN_L_time_array_train_{key}_{ind}.pt')
    torch.save(time_array_valid, train_args['save_dir'] + f'/Model/Allen/RNN_L_time_array_valid_{key}_{ind}.pt')

    return

def net_train_LC_L(train_iter, valid_iter, Classifier, spectro_args, train_args, key, ind, device):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Classifier.parameters(), train_args['lr'])
    # print('lr: ' + train_args['lr'])
    acu_array_train = []
    acu_array_valid = []
    time_array_train = []
    time_array_valid = []
    acu_valid_max = 0
    for epoch in range(0, train_args['epochs']):
        Classifier.train()
        acu_train = []
        epoch0 = 0
        time0_train = time.time()
        for x_train1, y_train, coordinate_train in train_iter:
            x_train = lfp_spectro(x_train1, spectro_args, train_args).squeeze(1).flatten(start_dim=1)
            y_train = y_train.to(device)
            py_train = Classifier(x_train.to(device))
            del x_train, x_train1
            if epoch0 % 800 == 0:
                print((torch.sum(torch.argmax(py_train, dim=1) == y_train) / y_train.size(0)).detach().cpu())
            L = loss_fn(py_train,y_train.to(device))
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            acu_train.append((torch.sum(torch.argmax(py_train, dim=1) == y_train) / y_train.size(0)).detach().cpu())
            epoch0 += 1
        print(f'Acu Train: {torch.mean(torch.tensor(acu_train))}')

        acu_array_train.append(torch.mean(torch.tensor(acu_train)))
        time_array_train.append(time.time() - time0_train)

        Classifier.eval()
        acu_valid = []
        time0_valid = time.time()
        for x_valid1, y_valid, coordinate_valid in valid_iter:
            x_valid = lfp_spectro(x_valid1, spectro_args, train_args).squeeze(1).flatten(start_dim=1)
            y_valid = y_valid.to(device)
            py_valid = Classifier(x_valid.to(device))
            del x_valid, x_valid1
            acu_valid.append((torch.sum(torch.argmax(py_valid, dim=1) == y_valid) / y_valid.size(0)).detach().cpu())

        print(f'Acu valid: {torch.mean(torch.tensor(acu_valid))}')
        # if torch.mean(torch.tensor(acu_train)) > train_args['overfitting_thres']:
        if acu_valid_max < torch.mean(torch.tensor(acu_valid)):
            torch.save(Classifier, train_args['save_dir'] + f'/Model/Allen/LC_L_Allen_{key}_{ind}.pth')
            torch.save(epoch, train_args['save_dir'] + f'/Model/Allen/LC_L_Allen_{key}_epoch{ind}.pt')
            acu_valid_max = torch.mean(torch.tensor(acu_valid))


        acu_array_valid.append(torch.mean(torch.tensor(acu_valid)))
        time_array_valid.append(time.time() - time0_valid)

    
    torch.save(torch.tensor(acu_array_train), train_args['save_dir'] + f'/Model/Allen/LC_L_Allen_{key}_train_acu{ind}.pt')
    torch.save(torch.tensor(acu_array_valid), train_args['save_dir'] + f'/Model/Allen/LC_L_Allen_{key}_valid_acu{ind}.pt')

    torch.save(time_array_train, train_args['save_dir'] + f'/Model/Allen/LC_L_time_array_train_{key}_{ind}.pt')
    torch.save(time_array_valid, train_args['save_dir'] + f'/Model/Allen/LC_L_time_array_valid_{key}_{ind}.pt')

    return
