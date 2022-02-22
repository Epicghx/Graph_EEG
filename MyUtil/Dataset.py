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
from MyUtil.data_preprocess import EEGprocess
import matplotlib.pyplot as plt

import time

#%% Dataset
        
class SEEDIVdataset(Dataset):
  
    def __init__(self, args, stage, index, session):
        global EEGdata, Sample_Num, EEGlabel
        self.args = args
        self.session = session   # the index of session order, [1,2,3]
        self.idx = index    # the index of trails selected as testset
        self.stage = stage  # "train" or "test"
        Process = EEGprocess(args)
        if args.sample_mode == "Series_process":
            EEGdata, EEGlabel, Sample_Num = Process.Series_process(session)
        elif args.sample_mode == "Sample_process":
            EEGdata, EEGlabel, Sample_Num = Process.Sample_process(session)

        testdata   = EEGdata[index * Sample_Num[-1][0] : (index+1) * Sample_Num[-1][0] ]
        testlabel  = EEGlabel[index * Sample_Num[-1][0] : (index+1) * Sample_Num[-1][0] ]
        traindata  = np.delete(EEGdata,  np.arange(index * Sample_Num[-1][0] , (index+1) * Sample_Num[-1][0]), 0)
        trainlabel = np.delete(EEGlabel, np.arange(index * Sample_Num[-1][0] , (index+1) * Sample_Num[-1][0]), 0)


        if stage == 'train':
            self.x = torch.from_numpy(traindata).float()
            self.y = torch.from_numpy(trainlabel).float()
            self.yprob = np.zeros((np.size(trainlabel, 0), 4))
            for i, label in enumerate(trainlabel):
                if label[0] == 0:
                    self.yprob[i, :] = [0.85, 0.05, 0.05, 0.05]
                elif label[0] == 1:
                    self.yprob[i, :] = [0.2 / 3, 1 - 0.2 / 3, 0.2 / 3, 0]
                elif label[0] == 2:
                    self.yprob[i, :] = [0.05, 0.05, 0.85, 0.05]
                elif label[0] == 3:
                    self.yprob[i, :] = [0.2 / 3, 0, 0.2 / 3, 1 - 0.2 / 3]

            self.yprob = torch.from_numpy(self.yprob).float()
        else:
            self.x = torch.from_numpy(testdata).float()
            self.y = torch.from_numpy(testlabel).float()

        
        tmp = sio.loadmat(args.data_path.format(dataset=args.dataset) +'initial_A_62x62.mat')
        self.initialA = torch.from_numpy(tmp['initial_A']).float()
        self.edge_index = torch.from_numpy(tmp['initial_weight_index']).long()

    def __len__(self):
        return np.size(self.y, 0)
 
    def __getitem__(self, idx):
        if self.stage == 'train':
            return Data(x=self.x[idx,:,:],y=self.y[idx,:].reshape(1,self.y.size(1)),yprob=self.yprob[idx,:].reshape(1,self.yprob.size(1)),
                    edge_index=self.edge_index)
        elif self.stage == 'test':
            return Data(x=self.x[idx, :, :], y=self.y[idx, :].reshape(1, self.y.size(1)), edge_index=self.edge_index)


# class SEEDIVdataset_test(Dataset):
#
#     def __init__(self, Dir='Data/'):
#         session = 1
#         # testdata = sio.loadmat(Dir+'Test_data{}.mat'.format(session))
#         testdata = np.load(Dir+'Test_data{}.npy'.format(session))
#         self.x = torch.from_numpy(testdata).float()
#         testlabel = np.load(Dir+'Test_label{}.npy'.format(session))
#         self.y = torch.from_numpy(testlabel).float()
#         tmp = sio.loadmat(Dir+'initial_A_62x62.mat')
#         self.edge_index = torch.from_numpy(tmp['initial_weight_index']).long()
#
#     def __len__(self):
#         return np.size(self.y, 0)
#
#     def __getitem__(self, idx):
#         return Data(x=self.x[idx,:,:],y=self.y[idx,:].reshape(1,self.y.size(1)),edge_index=self.edge_index)


def train_loader(args, stage, index, session, num_workers=8, pin_memory=True):


    train_dataset = SEEDIVdataset(args, stage, index, session)   #训练集标准化

    return DataLoader(train_dataset,
        batch_size=args.train_batch_size, shuffle=True, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)


def test_loader(args, stage, index, session, num_workers=4, pin_memory=False):


    test_dataset = SEEDIVdataset(args, stage, index, session)

    return DataLoader(test_dataset,
        batch_size=args.val_batch_size, shuffle=False, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)
