#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
"""

import torch
import torch.nn as nn

from utils import argparse_init

# from data.datasets_utils import random_rotate_neighs


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def create_feature_neighs(x, neigh_idxs):
    # The output of the function is BxNumxNeighsXF
    batch_size, num_points, num_features = x.shape
    num_neighs = neigh_idxs.shape[-1]
    x = x.transpose(1, 2)
    idx_base = torch.arange(0, batch_size, device=neigh_idxs.device).view(-1, 1, 1) * num_points

    idx = neigh_idxs + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(
        2, 1
    ).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, num_neighs, num_dims)
    return feature



def get_graph_feature(x, k, idx=None, only_intrinsic=False, permute_feature=True):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    else:
        if(len(idx.shape)==2):
            idx = idx.unsqueeze(0).repeat(batch_size,1,1)
        idx = idx[:, :, :k]
        k = min(k,idx.shape[-1])

    num_idx = idx.shape[1]

    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.contiguous()
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(
        2, 1
    ).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_idx, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    if only_intrinsic is True:
        feature = feature - x
    elif only_intrinsic == 'neighs':
        feature = feature
    elif only_intrinsic == 'concat':
        feature = torch.cat((feature, x), dim=3)
    else:
        feature = torch.cat((feature - x, x), dim=3)

    if permute_feature:
        feature = feature.permute(0, 3, 1, 2).contiguous()

    return feature



class DGCNN(nn.Module):
    def __init__(self, hparams, output_channels=40, latent_dim=None):
        super(DGCNN, self).__init__()


    @staticmethod
    def add_model_specific_args(parent_parser, task_name, dataset_name, is_lowest_leaf=False):
        parser = argparse_init.default_arg_parser(parents=[parent_parser], is_lowest_leaf=is_lowest_leaf)


        parser.add_argument(
            "--only_true_neighs", nargs="?", default=True, type=argparse_init.str2bool, const=True, help="Use the grpah neightborhood in all dgcnn steps or only at the first iteration",
        )
        parser.add_argument(
            "--use_inv_features", nargs="?", default=False, type=argparse_init.str2bool, const=True, help="Evaluate sensetivity to noise",
        )
        parser.add_argument("--concat_xyz_to_inv", nargs="?", default=False, type=argparse_init.str2bool, const=True,)

        parser.add_argument(
            "--DGCNN_latent_dim",
            type=int,
            default=512,
        )
        parser.add_argument("--bb_size", default=32, type=int, help="the building block size")
        parser.add_argument("--nn_depth", default=4, type=int, help="num of convs")

        return parser

