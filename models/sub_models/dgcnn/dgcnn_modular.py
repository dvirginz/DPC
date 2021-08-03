#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
"""

from torch_cluster import knn
from models.sub_models.dgcnn.dgcnn import get_graph_feature

import torch
import torch.nn as nn
import torch.nn.functional as F



class DGCNN_MODULAR(nn.Module):
    def __init__(self, hparams, output_dim=None,use_inv_features=False,latent_dim=None,use_only_classification_head=False,): # bb - Building block
        super(DGCNN_MODULAR, self).__init__()
        self.hparams = hparams
        self.num_neighs = hparams.num_neighs
        self.latent_dim = latent_dim if latent_dim is not None else hparams.DGCNN_latent_dim
        self.input_features = self.hparams.in_features_dim * 2
        self.use_inv_features = use_inv_features
        if(self.use_inv_features):
            self.input_features = 4

            if(self.hparams.use_sprin):
                self.input_features = 8
            if(self.hparams.concat_xyz_to_inv):
                self.input_features += 3
        self.depth = self.hparams.nn_depth
        bb_size = self.hparams.bb_size
        output_dim = output_dim if output_dim is not None else self.hparams.latent_dim
        if(not use_only_classification_head):
            self.convs = []
            for i in range(self.depth):
                in_features = self.input_features if i == 0 else bb_size * (2 ** (i+1)) * 2
                out_features = bb_size * 4 if i == 0 else in_features
                self.convs.append(nn.Sequential(
                    nn.Conv2d(in_features, out_features, kernel_size=1, bias=False), nn.BatchNorm2d(out_features), nn.LeakyReLU(negative_slope=0.2),
                )
            )
            last_in_dim = bb_size * 2 * sum([2 ** i for i in range(1,self.depth + 1,1)])
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(last_in_dim, self.latent_dim, kernel_size=1, bias=False), nn.BatchNorm1d(self.latent_dim), nn.LeakyReLU(negative_slope=0.2),
                )
            )
            self.convs = nn.ModuleList(self.convs)

        input_latent_dim = self.latent_dim if not use_only_classification_head else self.hparams.DGCNN_latent_dim
        self.linear1 = nn.Linear(input_latent_dim * 2, bb_size * 64, bias=False)
        self.bn6 = nn.BatchNorm1d(bb_size * 64)
        self.dp1 = nn.Dropout(p=self.hparams.dropout)

        self.linear2 = nn.Linear(bb_size * 64, bb_size * 32)
        self.bn7 = nn.BatchNorm1d(bb_size * 32)
        self.dp2 = nn.Dropout(p=self.hparams.dropout)

        self.linear3 = nn.Linear(bb_size * 32, output_dim)

    def forward_per_point(self, x, start_neighs=None):
        x = x.transpose(1, 2)  # DGCNN assumes BxFxN

        if(start_neighs is None):
            start_neighs = knn(x,k=self.num_neighs)
        
        x = get_graph_feature(x, k=self.num_neighs, idx=start_neighs, only_intrinsic=False if not self.use_inv_features else 'concat')#only_intrinsic=self.hparams.only_intrinsic)
        other = x[:,:3,:,:]

        if(self.hparams.concat_xyz_to_inv):
            x = torch.cat([x,other],dim=1)

        outs = [x]
        for conv in self.convs[:-1]:
            if(len(outs) > 1):
                x = get_graph_feature(outs[-1], k=self.num_neighs, idx=None if not self.hparams.only_true_neighs else start_neighs)
            x = conv(x)
            outs.append(x.max(dim=-1, keepdim=False)[0])

        x = torch.cat(outs[1:], dim=1)
        features = self.convs[-1](x)
        return features.transpose(1,2)
        # It is advised
    
    def aggregate_all_points(self,features_per_point):
        if(features_per_point.shape[1] == self.hparams.num_points):
            features_per_point = features_per_point.transpose(1,2)
        batch_size = features_per_point.size(0)
        x1 = features_per_point.max(-1)[0].view(batch_size, -1)
        x2 = features_per_point.mean(-1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x


    def forward(self, x, start_neighs, sigmoid_for_classification=False):

        features_per_point = self.forward_per_point(x, start_neighs=start_neighs)
        features_aggregated = self.aggregate_all_points(features_per_point)
        if sigmoid_for_classification:
            features_aggregated = torch.sigmoid(features_aggregated)

        return features_aggregated.squeeze(-1),features_per_point.transpose(1,2) # conv assumes B F N

