# -*- coding: utf-8 -*-
"""
Created on Mon May 11 22:06:45 2020

@author: Miz
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.nn import SGConv, global_add_pool
from torch_scatter import scatter_add
from torch_geometric.data import Data, Dataset, DataLoader, InMemoryDataset

import numpy as np
import scipy.io as sio
import scipy
from scipy.sparse import csr_matrix
from MyUtil.data_preprocess import EEGprocess
from MyUtil.Electrodes_62 import Electrodes
import matplotlib.pyplot as plt

import time

#%% Dataset
        
class SDdataset(Dataset):
  
    def __init__(self, args, stage, subject, session):
        global EEGdata, Sample_Num, EEGlabel
        self.args = args
        self.session = session   # the index of session order, [1,2,3]
        self.idx = subject    # the index of trails selected as testset
        self.stage = stage  # "train" or "test"
        Process = EEGprocess(args)

        # if args.sample_mode == "Series_process":
        #     EEGdata, EEGlabel, Sample_Num = Process.Series_process(session)
        # elif args.sample_mode == "Sample_process":
        #     EEGdata, EEGlabel, Sample_Num = Process.Sample_process(session)

        if args.train_pattern == 'SD':
            traindata, testdata, trainlabel, testlabel, adj_freq = Process.SD_process(subject, session)
        else:
            testdata   = EEGdata[subject * Sample_Num[-1][0] : (subject+1) * Sample_Num[-1][0] ]
            testlabel  = EEGlabel[subject * Sample_Num[-1][0] : (subject+1) * Sample_Num[-1][0] ]
            traindata  = np.delete(EEGdata,  np.arange(subject * Sample_Num[-1][0] , (subject+1) * Sample_Num[-1][0]), 0)
            trainlabel = np.delete(EEGlabel, np.arange(subject * Sample_Num[-1][0] , (subject+1) * Sample_Num[-1][0]), 0)

        epsilon = 0.2
        if stage == 'train':
            self.x = torch.from_numpy(traindata).float()
            self.y = torch.from_numpy(trainlabel).float()
            if args.dataset == 'SEEDIV':
                self.yprob = np.zeros((trainlabel.shape[0], 4))
                for i, label in enumerate(trainlabel):
                    if label[0] == 0:
                        self.yprob[i, :] = [1-(3*epsilon/4), epsilon/4, epsilon/4, epsilon/4]
                    elif label[0] == 1:
                        self.yprob[i, :] = [epsilon/3, 1-(2*epsilon/3), epsilon/3, 0]
                    elif label[0] == 2:
                        self.yprob[i, :] = [epsilon/4, epsilon/4, 1-(3*epsilon/4), epsilon/4]
                    elif label[0] == 3:
                        self.yprob[i, :] = [epsilon/3, 0, epsilon/3, 1-(2*epsilon/3)]

                self.yprob = torch.from_numpy(self.yprob).float()
            elif args.dataset == 'SEED':
                self.yprob = np.zeros((trainlabel.shape[0], 3))
                for i, label in enumerate(trainlabel):
                    if label[0] == 0:
                        self.yprob[i, :] = [1-(2*epsilon/3), 2*epsilon/3, 0]
                    elif label[0] == 1:
                        self.yprob[i, :] = [epsilon/3, 1-(2*epsilon/3), epsilon/3]
                    elif label[0] == 2:
                        self.yprob[i, :] = [0, 2*epsilon/3, 1-(2*epsilon/3)]
                self.yprob = torch.from_numpy(self.yprob).float()
        elif stage == 'test':
            self.x = torch.from_numpy(testdata).float()
            self.y = torch.from_numpy(testlabel).float()

        # elec = Electrodes(add_global_connections=True, expand_3d=False)

        tmp = sio.loadmat(args.data_path.format(dataset=args.dataset) +'initial_A_62x62.mat')
        self.adj = torch.from_numpy(tmp['initial_A']).float()
        self.edge_index = torch.from_numpy(tmp['initial_weight_index']).long()
        # self.adj[:, :] = 0.5
        # self.adj = self.adj + 0.5 * torch.eye(self.adj.size(0))
        if args.arch == 'GCN_Net':
            self.adj[:, :] = 0.5

        # self.adj = torch.from_numpy(elec.adjacency_matrix).float()
        # self.adj = torch.from_numpy(adj_freq).type(torch.FloatTensor)
        # row = torch.from_numpy(np.arange(62).repeat(62))
        # col = torch.from_numpy(np.tile(np.arange(62), 62))
        # self.edge_index = torch.stack((row, col), 0)

    def __len__(self):
        return np.size(self.y, 0)
 

 
    def __getitem__(self, idx):
        if self.stage == 'train':
            return Data(x=self.x[idx,:,:],y=self.y[idx,:].reshape(1,self.y.size(1)),yprob=self.yprob[idx,:].reshape(1,self.yprob.size(1)),
                        edge_index=self.edge_index)
        elif self.stage == 'test':
            return Data(x=self.x[idx, :, :], y=self.y[idx, :].reshape(1, self.y.size(1)), edge_index=self.edge_index)

    # @classmethod
    # def build_graph(cls):
    #
    #     def adjacency():
    #         row_ = np.array(
    #             [0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
    #              13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 23, 23, 24, 24, 25, 25, 26, 26,
    #              27, 27, 28, 28, 29, 29, 30, 30, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40,
    #              41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54,
    #              54, 55, 55, 56, 57, 58, 59,
    #              60, 1, 3, 2, 3, 4, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 6, 14, 7, 15, 8, 16, 9, 17, 10, 18, 11, 19, 12,
    #              20, 13, 21, 22, 15, 23, 16, 24, 17, 25, 18, 26, 19, 27, 20, 28, 21, 29, 22, 30, 31, 24, 32, 25, 33, 26,
    #              34, 27, 35, 28, 36, 29, 37, 30, 38, 31, 39, 40, 33, 41, 34, 42, 35, 43, 36, 44, 37, 45, 38, 46, 39, 47,
    #              40, 48, 49, 42, 50, 43, 51, 44, 52, 45, 52, 46, 53, 47, 54, 48, 54, 49, 55, 56, 51, 57, 52, 57, 53, 58,
    #              54, 59, 55, 60, 56, 61, 61, 58, 59, 60, 61])
    #
    #         col_ = np.array(
    #             [1, 3, 2, 3, 4, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 6, 14, 7, 15, 8, 16, 9, 17, 10, 18, 11, 19, 12, 20,
    #              13, 21, 22, 15, 23, 16, 24, 17, 25, 18, 26, 19, 27, 20, 28, 21, 29, 22, 30, 31, 24, 32, 25, 33, 26, 34,
    #              27, 35, 28, 36, 29, 37, 30, 38, 31, 39, 40, 33, 41, 34, 42, 35, 43, 36, 44, 37, 45, 38, 46, 39, 47, 40,
    #              48, 49, 42, 50, 43, 51, 44, 52, 45, 52, 46, 53, 47, 54, 48, 54, 49, 55, 56, 51, 57, 52, 57, 53, 58, 54,
    #              59, 55, 60, 56, 61, 61, 58,
    #              59, 60, 61, 0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11,
    #              11, 12, 12, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 23, 23, 24, 24, 25,
    #              25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38,
    #              39, 39, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 50, 50, 51, 51, 52, 52,
    #              53, 53, 54, 54, 55, 55, 56, 57, 58, 59, 60])
    #         data_ = np.ones(236).astype('float32')
    #         A = scipy.sparse.csr_matrix((data_, (row_, col_)), shape=(62, 62))
    #         return A
    #
    #     adj = adjacency()
    #     return adj


def train_loader(args, stage, index, session, num_workers=8, pin_memory=True):


    train_dataset = SDdataset(args, stage, index, session)   #训练集标准化

    return DataLoader(train_dataset,
        batch_size=args.train_batch_size, shuffle=True, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)


def test_loader(args, stage, index, session, num_workers=4, pin_memory=False):


    test_dataset = SDdataset(args, stage, index, session)

    return DataLoader(test_dataset,
        batch_size=args.val_batch_size, shuffle=False, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)
