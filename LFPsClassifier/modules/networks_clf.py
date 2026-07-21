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


# @title Define AnyNet_L
def init_cnn(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)

class ResNeXtBlock(nn.Module):
    """The ResNeXt block."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False, strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1, stride=1)
        self.conv2 = nn.LazyConv2d(bot_channels, kernel_size=3, stride=strides,
                                   padding=1, groups=bot_channels//groups)
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=1)
        self.bn1 = nn.LazyBatchNorm2d(momentum=0.5)
        self.bn2 = nn.LazyBatchNorm2d(momentum=0.5)
        self.bn3 = nn.LazyBatchNorm2d(momentum=0.5)
        if use_1x1conv:
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)
            self.bn4 = nn.LazyBatchNorm2d(momentum=0.5)
        else:
            self.conv4 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return F.relu(Y + X)

class Frequency_scaling(nn.Module):
    def __init__(self, frequency_bin):
        super().__init__()
        self.scaling_layer = nn.LazyLinear(frequency_bin)

    def forward(self, X):
        return self.scaling_layer(X.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

class AnyNet_L(nn.Module):
    def stem(self, num_channels):
        return nn.Sequential(nn.LazyConv2d(num_channels, kernel_size=3, stride=2, padding=1),
                             nn.LazyBatchNorm2d(momentum=0.5), nn.ReLU())

    def stage(self, depth, num_channels, groups, bot_mul):
        blk = []
        for i in range(depth):
            if i == 0:
                blk.append(ResNeXtBlock(num_channels, groups, bot_mul, use_1x1conv=True, strides=2))
            else:
                blk.append(ResNeXtBlock(num_channels, groups, bot_mul))
        return nn.Sequential(*blk)

    def __init__(self, arch, stem_channels, frequency_bin, num_classes=10):
        super().__init__()
        self.frequency_scaling = Frequency_scaling
        self.net = nn.Sequential(self.frequency_scaling(frequency_bin), self.stem(stem_channels))
        for i, s in enumerate(arch):
            if not type(s) == tuple:
                self.net.add_module(f'stage{i+1}', self.stage(*arch))
                break
            else:
                self.net.add_module(f'stage{i+1}', self.stage(*s))
        self.net.add_module('head', nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                                                  nn.LazyLinear(num_classes)))

    def forward(self, x):
        return self.net(x)


################################################################################
# ViT ##########################################################################
################################################################################

class Frequency_scaling(nn.Module):
    def __init__(self, frequency_bin):
        super().__init__()
        self.scaling_layer = nn.LazyLinear(frequency_bin)

    def forward(self, X):
        return self.scaling_layer(X.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

def masked_softmax(X, valid_lens):
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
    X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
    return F.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):
    def __init__(self, dropout, num_heads=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads # To be covered later

    def forward(self, queries, keys, values, valid_lens=None, window_mask=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        if window_mask is not None:
            num_windows = window_mask.shape[0]
            n, num_queries, num_kv_pairs = scores.shape
            scores = scores.reshape((n // (num_windows * self.num_heads), num_windows, self.num_heads, num_queries, num_kv_pairs)) + window_mask.unsqueeze(1).unsqueeze(0)
            scores = scores.reshape((n, num_queries, num_kv_pairs))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout, num_heads)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

    def transpose_qkv(self, X):
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, queries, keys, values, valid_lens, window_mask=None):
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        output = self.attention(queries, keys, values, valid_lens, window_mask)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,stride=patch_size)

    def forward(self, X):
        return self.conv(X).flatten(2).transpose(1, 2)

class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(self.dense1(x)))))

class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens,num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)
    def forward(self, X, valid_lens=None):
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))

class ViT_L(nn.Module):
    def __init__(self, frequency_bin, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                num_heads, num_blks, emb_dropout, blk_dropout,
                use_bias=False, num_classes=10):
        super().__init__()
        self.frequency_scaling = Frequency_scaling(frequency_bin)
        self.patch_embedding = PatchEmbedding(img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1 # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(num_hiddens, num_hiddens, mlp_num_hiddens,
                                                    num_heads, blk_dropout, use_bias))
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens), nn.Linear(num_hiddens, num_classes))

    def forward(self, X):
        X = self.frequency_scaling(X)
        X = self.patch_embedding(X)
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])

################################################################################
# RNN ##########################################################################
################################################################################
class RNN_Frequency_scaling(nn.Module):
    def __init__(self, frequency_bin):
        super().__init__()
        self.scaling_layer = nn.LazyLinear(frequency_bin)

    def forward(self, X):
        return self.scaling_layer(X)

class RNN_L(nn.Module):
    def __init__(self, frequency_bin, RNN_args):
        super().__init__()
        self.frequency_scaling = RNN_Frequency_scaling(frequency_bin)
        self.GRU = nn.Sequential(
            nn.GRU(RNN_args['input_size'], RNN_args['hidden_size'], RNN_args['num_layers'], batch_first = True),
        )
        self.FC = nn.Sequential(
            nn.Flatten(start_dim = 1),
            nn.LazyLinear(out_features = RNN_args['category_num']),
        )

    def forward(self, x):
        x = self.frequency_scaling(x)
        o, hn = self.GRU(x)
        return self.FC(o[:,-1,:])


################################################################################
# Linear Classifier ############################################################
################################################################################
class LinearClassifier(nn.Module):
    def __init__(self, frequency_bin, LC_args):
        super().__init__()
        self.FC = nn.Sequential(
            nn.LazyLinear(out_features = LC_args['category_num']),
        )

    def forward(self, x):
        return self.FC(x)