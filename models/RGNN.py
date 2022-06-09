#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:19:22 2020

@author: miz
"""
import ipdb
import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.nn import SGConv, global_add_pool
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

import numpy as np

import matplotlib.pyplot as plt


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


# def add_remaining_self_loops(edge_index,
#                              edge_weight=None,
#                              fill_value=1,
#                              num_nodes=None):
#     num_nodes = maybe_num_nodes(edge_index, num_nodes)
#     row, col = edge_index
#
#     mask = row != col   # non-diag
#     inv_mask = ~mask    # diag
#     loop_weight = torch.full(
#         (num_nodes, ),
#         fill_value,
#         dtype=None if edge_weight is None else edge_weight.dtype,
#         device=edge_index.device)
#
#     if edge_weight is not None:
#         assert edge_weight.numel() == edge_index.size(1)
#         remaining_edge_weight = edge_weight[inv_mask]   # diag value
#         if remaining_edge_weight.numel() > 0:
#             loop_weight[row[inv_mask]] = remaining_edge_weight
#         edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)  #拼接，diag数据在末尾992个
#
#     loop_index = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)
#     loop_index = loop_index.unsqueeze(0).repeat(2, 1)
#     edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)
#
#     return edge_index, edge_weight


class NewSGConv(SGConv):
    def __init__(self, num_features, num_classes, K=1, cached=False, bias=True):
        super(NewSGConv, self).__init__(num_features, num_classes, K=K, cached=cached, bias=bias)

    # allow negative edge weights
    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index
        deg = scatter_add(torch.abs(edge_weight), row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        alpha = 0.10
        if not self.cached or self.cached_result is None:
            edge_index, norm = NewSGConv.norm(
                edge_index, x.size(0), edge_weight, dtype=x.dtype)
            emb = alpha * x
            for k in range(self.K):
                x = self.propagate(edge_index, x=x, norm=norm)
                emb = emb + (1 - alpha) * x / self.K
            self.cached_result = emb
        return self.lin(self.cached_result)

    def message(self, x_j, norm):
        # x_j: (batch_size*num_nodes*num_nodes, num_features)
        # norm: (batch_size*num_nodes*num_nodes, )
        # ipdb.set_trace()
        return norm.view(-1, 1) * x_j


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class SymSimGCNNet(torch.nn.Module):
    def __init__(self, num_nodes, learn_edge_weight, edge_weight, num_features, num_hiddens, num_classes, K, dropout=0.5, domain_adaptation=""):
        """
            num_nodes: number of nodes in the graph
            learn_edge_weight: if True, the edge_weight is learnable
            edge_weight: initial edge matrix
            num_features: feature dim for each node/channel
            num_hiddens: a tuple of hidden dimensions
            num_classes: number of emotion classes
            K: number of layers
            dropout: dropout rate in final linear layer
            domain_adaptation: RevGrad
        """
        super(SymSimGCNNet, self).__init__()
        self.domain_adaptation = domain_adaptation
        self.num_nodes = num_nodes
        self.xs, self.ys = torch.tril_indices(self.num_nodes, self.num_nodes, offset=0)
        edge_weight = edge_weight.reshape(self.num_nodes, self.num_nodes)[self.xs, self.ys] # strict lower triangular values
        self.edge_weight = nn.Parameter(edge_weight, requires_grad=learn_edge_weight)
        self.dropout = dropout
        self.conv1 = NewSGConv(num_features=num_features, num_classes=num_hiddens[0], K=K)
        self.fc = nn.Linear(num_hiddens[0], num_classes)
        if self.domain_adaptation in ["RevGrad"]:
            self.domain_classifier = nn.Linear(num_hiddens[0], 2)

    def forward(self, data, alpha=0):
        batch_size = len(data.y)
        x, edge_index = data.x, data.edge_index   #edge_index: 2x16x62x62
        edge_weight = torch.zeros((self.num_nodes, self.num_nodes), device=edge_index.device)
        edge_weight[self.xs.to(edge_weight.device), self.ys.to(edge_weight.device)] = self.edge_weight
        edge_weight = edge_weight + edge_weight.transpose(1,0) - torch.diag(edge_weight.diagonal()) # copy values from lower tri to upper tri
        edge_weight = edge_weight.reshape(-1).repeat(batch_size)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        # domain classification
        domain_output = None
        if self.domain_adaptation in ["RevGrad"]:
            reverse_x = ReverseLayerF.apply(x, alpha)
            domain_output = self.domain_classifier(reverse_x)
        x = global_add_pool(x, data.batch, size=batch_size)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.softmax(self.fc(x), dim=0)
        x = self.fc(x)
        return x, domain_output

def RGNN( num_nodes, learn_edge_weight, edge_weight, num_features, num_hiddens, num_classes, K, dropout, mode):

    return SymSimGCNNet(num_nodes, learn_edge_weight, edge_weight, num_features, num_hiddens, num_classes, K, dropout, mode)


def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit, dim=1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=1)
                                   - F.log_softmax(q_logit, dim=1)), 1)
    return torch.mean(_kl)



def confusion_matrix(preds, labels, conf_matrix):
    # preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

conf_matrix = torch.zeros(4, 4)



# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# plot_confusion_matrix(confusion_matrix(o,l,conf_matrix), classes=['0','1','2','3'],normalize=True)

