from __future__ import print_function
from __future__ import division

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# import libraries
import sys
import pickle as pkl

import ipdb
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy import stats
import torch
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from torch_scatter import scatter_add
from torch_geometric.nn import SGConv, global_add_pool
# from torch_geometric.utils import add_remaining_self_loops


# torch.cuda.device_count()


# seed = 5
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        # super(GraphConvolution, self).__init__(in_features, out_features, K=1, cached=cached, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        # self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.W = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.lin.weight.size(1))
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        # torch.nn.init.normal_(tensor=self.W.weight, std=stdv)
        torch.nn.init.normal_(tensor=self.W.weight, std=stdv)
        # self.W.weight.data.uniform_(-stdv, stdv)
        # self.W.bias.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        # if self.lin.bias is not None:
        #     self.lin.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, batch_size, adj=None):
        # support = torch.mm(input, self.weight)
        # output = torch.mm(adj, support)
        support = self.W(x)          # input in GDC means raw features, AWX
        output = torch.matmul(adj, support.reshape(batch_size, adj.size(0), -1)).reshape(x.size(0), -1)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        # alpha = 0.10
        # if not self.cached or self.cached_result is None:
        #     # edge_index, norm = SGC.norm(
        #     #     edge_index, x.size(0), edge_weight, dtype=x.dtype)
        #     # emb = alpha * x
        #     for k in range(self.K):
        #         # ipdb.set_trace()
        #         x = self.propagate(edge_index, x=x, norm=edge_weight)
        #         # emb = emb + (1 - alpha) * x / self.K
        #     # self.cached_result = emb
        #     self.cached = x
        #     output = self.lin(self.cached_result)
        #     # ipdb.set_trace()
        #     # if self.lin.bias is not None:
        #     #     return output + self.lin.bias
        #     # else:
        #     return output


    # def message(self, x_j, norm):
    #     # x_j: (batch_size*num_nodes*num_nodes, num_features)
    #     # norm: (batch_size*num_nodes*num_nodes, )
    #     # ipdb.set_trace()
    #     return norm.view(-1, 1) * x_j

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class BBGDC(nn.Module):
    def __init__(self, num_pars, alpha=0.8, kl_scale=1.0):
        super(BBGDC, self).__init__()
        self.num_pars = num_pars
        self.alpha = alpha
        self.kl_scale = kl_scale
        self.a_uc = nn.Parameter(torch.FloatTensor(self.num_pars))
        self.b_uc = nn.Parameter(torch.FloatTensor(self.num_pars))
        self.a_uc.data.uniform_(1.0, 1.5)
        self.b_uc.data.uniform_(0.49, 0.51)

    def get_params(self):
        a = F.softplus(self.a_uc.clamp(min=-10.))  # ln(1+e^x)
        b = F.softplus(self.b_uc.clamp(min=-10., max=50.))
        return a, b

    def sample_pi(self):
        a, b = self.get_params()
        u = torch.rand(self.num_pars).clamp(1e-6, 1-1e-6)
        if torch.cuda.is_available():
            u = u.cuda()
        return (1 - u.pow_(1. / b)).pow_(1. / a)

    def get_weight(self, num_samps, training, samp_type='rel_ber'):
        temp = torch.Tensor([0.67])
        if torch.cuda.is_available():
            temp = temp.cuda()

        if training:
            pi = self.sample_pi()
            p_z = RelaxedBernoulli(temp, probs=pi)
            z = p_z.rsample(torch.Size([num_samps]))
        else:
            if samp_type == 'rel_ber':
                pi = self.sample_pi()
                p_z = RelaxedBernoulli(temp, probs=pi)
                z = p_z.rsample(torch.Size([num_samps]))
            elif samp_type == 'ber':
                pi = self.sample_pi()
                p_z = torch.distributions.Bernoulli(probs=pi)
                z = p_z.sample(torch.Size([num_samps]))
        return z, pi

    def get_reg(self):
        a, b = self.get_params()
        kld = (1 - self.alpha / a) * (-0.577215664901532 - torch.digamma(b) - 1. / b) + torch.log(
            a * b + 1e-10) - math.log(self.alpha) - (b - 1) / b
        kld = (self.kl_scale) * kld.sum()
        return kld

class GDC(nn.Module):
    def __init__(self, nfeat_list, adj, dropout, nblock, nlay, num_nodes):
        super(GDC, self).__init__()

        assert len(nfeat_list) == nlay + 1
        self.nlay = nlay
        self.nblock = nblock
        self.num_nodes = num_nodes
        self.num_edges = int(pow(num_nodes, 2))
        self.drpcon_list = []
        self.dropout = dropout

        self.xs, self.ys = torch.tril_indices(self.num_nodes, self.num_nodes, offset=0)
        edge_weight = adj.reshape(self.num_nodes, self.num_nodes)[
            self.xs, self.ys]  # strict lower triangular values
        self.edge_weight = nn.Parameter(edge_weight, requires_grad=True)

        gcs_list = []
        idx = 0
        for i in range(nlay):
            if i == 0:
                self.drpcon_list.append(BBGDC(1))
                gcs_list.append([str(idx), GraphConvolution(nfeat_list[i], nfeat_list[i + 1])])
                idx += 1
            else:
                self.drpcon_list.append(BBGDC(1))
                for j in range(self.nblock):
                    gcs_list.append([str(idx), GraphConvolution(int(nfeat_list[i] / self.nblock), nfeat_list[i + 1])])
                    idx += 1

        self.drpcons = nn.ModuleList(self.drpcon_list)
        self.gcs = nn.ModuleDict(gcs_list)
        self.nfeat_list = nfeat_list

    @staticmethod
    def norm(edge_weight):
        deg = abs(edge_weight).sum(1)
        d_inv_sqrt = deg.pow(-0.5)
        d_inv_sqrt[d_inv_sqrt == float('inf')] = 0
        d_inv_sqrt[d_inv_sqrt == float('nan')] = 0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        ipdb.set_trace()
        return d_mat_inv_sqrt.mm(edge_weight).mm(d_mat_inv_sqrt)

    def forward(self, data, training=True, mul_type='norm_first', samp_type='rel_ber'):
        x = data.x
        batch_size = len(data.y)

        edge_weight = torch.zeros((self.num_nodes, self.num_nodes), device=data.edge_index.device)
        edge_weight[self.xs.to(edge_weight.device), self.ys.to(edge_weight.device)] = self.edge_weight
        edge_weight = edge_weight + edge_weight.transpose(1, 0) - torch.diag(
            edge_weight.diagonal())  # copy values from lower tri to upper tri

        kld_loss = 0.0
        drop_rates = []
        for i in range(self.nlay):
            # ipdb.set_trace()
            mask_vec, drop_prob = self.drpcons[i].get_weight(self.nblock * self.num_edges, training, samp_type)
            mask_vec = torch.squeeze(mask_vec)
            drop_rates.append(drop_prob)
            if i == 0:
                mask_mat = torch.reshape(mask_vec[:self.num_edges], (self.num_nodes, self.num_nodes)).cuda()

                if mul_type == 'norm_sec':
                    # adj_lay = normalize_torch(torch.mul(mask_mat, adj) + torch.eye(adj.shape[0]).cuda())
                    adj_lay = self.norm(edge_weight)
                elif mul_type == 'norm_first':
                    adj = self.norm(edge_weight)
                    adj_lay = torch.mul(mask_mat, adj).cuda()
                ipdb.set_trace()
                x = F.relu(self.gcs[str(i)](x, batch_size, adj_lay))
                # x = self.gcs[str(i)](x, batch_size, adj_lay)
                x = F.dropout(x, self.dropout, training=training)

            else:
                feat_pblock = int(self.nfeat_list[i] / self.nblock)
                for j in range(self.nblock):
                    mask_mat = torch.reshape(mask_vec[j * self.num_edges:(j + 1) * self.num_edges]
                                             , (self.num_nodes, self.num_nodes)).cuda()

                    if mul_type == 'norm_sec':
                        # adj_lay = normalize_torch(torch.mul(mask_mat, adj) + torch.eye(adj.shape[0]).cuda())
                        adj_lay = self.norm(edge_weight)
                    elif mul_type == 'norm_first':
                        adj = self.norm(edge_weight)
                        adj_lay = torch.mul(mask_mat, adj).cuda()

                    if i < (self.nlay - 1):
                        if j == 0:
                            x_out = self.gcs[str((i - 1) * self.nblock + j + 1)](
                                x[:, j * feat_pblock:(j + 1) * feat_pblock], batch_size, adj_lay)
                        else:
                            x_out = x_out + self.gcs[str((i - 1) * self.nblock + j + 1)](
                                x[:, j * feat_pblock:(j + 1) * feat_pblock], batch_size, adj_lay)
                    else:
                        if j == 0:
                            out = self.gcs[str((i - 1) * self.nblock + j + 1)](
                                x[:, j * feat_pblock:(j + 1) * feat_pblock], batch_size, adj_lay)
                        else:
                            out = out + self.gcs[str((i - 1) * self.nblock + j + 1)](
                                x[:, j * feat_pblock:(j + 1) * feat_pblock], batch_size, adj_lay)

                if i < (self.nlay - 1):
                    x = x_out
                    x = F.dropout(x, self.dropout, training=training)
                    # x = F.dropout(F.relu(x), self.dropout, training=training)

            kld_loss += self.drpcons[i].get_reg()
        output = global_add_pool(out, data.batch, size=batch_size)
        # nll_loss = self.loss(labels, output, obs_idx)
        # tot_loss = nll_loss + warm_up * kld_loss
        drop_rates = torch.stack(drop_rates)

        return output, kld_loss, drop_rates

    def loss(self, labels, preds, obs_idx):
        return F.nll_loss(preds[obs_idx], labels[obs_idx])
