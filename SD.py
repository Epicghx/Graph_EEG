#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Jan 28 15:16 2021

@author: epic
"""
import os

import ipdb
import torch
import logging
import argparse
import itertools
import warnings
import random
import time

import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard

import MyUtil.Dataset as dataset
from MyUtil.manager import Manager
import models

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

## initial parameters
device = torch.device('cuda:0')

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()

parser.add_argument('--arch', type=str, default='RGNN',
                   help='Architectures')
parser.add_argument('--num_classes', type=int, default=3,
                   help='Num outputs for dataset')
parser.add_argument('--lr', type=float, default=5e-3,
                   help='Learning rate for parameters, used for baselines')
parser.add_argument('--train_pattern', type=str, default='SD',
                    choices=['SD', 'SI'],
                   help='subject_dependent or subject independent')

# Paths.
parser.add_argument('--dataset', type=str, default='SEED',
                    help='Name of dataset')
parser.add_argument('--data_path', type=str, default='Data/{dataset}/',
                    help='Path of dataset')
parser.add_argument('--num_layer', type=int, default=2,
                   help='Num layers of GCN')

# Universal parameters

parser.add_argument('--weight_decay', type=float, default=5e-4,
                   help='Weight decay for parameters, used for baselines')
parser.add_argument('-b', '--train_batch_size', type=int, default=16,
                   help='input batch size for training')
parser.add_argument('-vb', '--val_batch_size', type=int, default=16,
                   help='input batch size for validation')
parser.add_argument('--workers', type=int, default=8, help='')
parser.add_argument('--dropout', type=float, default=0.7,
                   help='Dropout rate')
parser.add_argument('--channel', type=float, default=62,
                   help='channel numbers of features')

# Optimization options.
parser.add_argument('--feature', type=str, default='de_LDS{}',
                    choices=['de_LDS{}', 'PSD{}'],
                   help='features used in GNN')
parser.add_argument('--freq_num', type=int, default=5,
                   help='number of freq bands used')
parser.add_argument('--max_size', type=int, default=128, help='the maximum length in all trails')   #64 for SEEDIV   265 for SEED
parser.add_argument('--mode', type=str, default='',
                    choices=['RevGrad', ''],
                   help='Gan mode')
parser.add_argument('--sample_mode', type=str, default='Sample_process',
                    choices=['Sample_process', 'Series_process'],
                   help='mode of input data processing')
parser.add_argument('--data_norm', action='store_true', default=False,
                   help='data normalization')

# Other.
parser.add_argument('--cuda', action='store_true', default=True,
                   help='use CUDA')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--checkpoint_format', type=str,
                    default='./{save_folder}/checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--h', type=int, default=128,
                    help='hidden layer channel')

#}}}


def main(args, subject, session):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    global feature_dim

    train_loader = dataset.train_loader(args, 'train', index=subject,
                                        session=session)  # index means trails order [0,24] & [0,15]
    test_loader = dataset.test_loader(args, 'test', index=subject,
                                      session=session)  # session means 3 sessions (data dir name)
    print("---------------Dataset information---------------")
    print('Session:{}-------Subject: {}'.format(session, subject))
    print('Train data shape is {}'.format(train_loader.dataset.x.shape))
    print('Test  data shape is {}'.format(test_loader.dataset.x.shape))
    print("-------------------------------------------------")

    if args.sample_mode == "Sample_process":
        feature_dim = args.freq_num
    elif args.sample_mode == "Series_process":
        feature_dim = args.max_size * args.freq_num

    model = models.__dict__[args.arch](args.channel, True, train_loader.dataset.adj,
                                       feature_dim, [args.h], args.num_classes, args.num_layer, args.dropout, args.mode)

    model.to(device)

    manager = Manager(args, model, train_loader, test_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  #decay = 0
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    best_acc = 0
    for epoch_idx in range(args.epochs):
        # writer.add_histogram('sgc', model.conv1.lin.weight.reshape(-1))
        # writer.add_histogram('edge', model.edge_weight)
        # writer.add_histogram('fc', model.fc.weight)
        avg_train_acccuracy = manager.train(optimizer, epoch_idx)
        avg_test_accuracy   = manager.eval(epoch_idx)
        # scheduler.step()
        if avg_test_accuracy >= best_acc:
            best_acc = avg_test_accuracy
            best_epoch = epoch_idx+1
            print('Current epoch is {}, Best eval Acc is {:.4f}'.format(best_epoch, best_acc))
        #     es = 0
        # else:
        #     es += 1
        #     if es > 30:
        #         print("Early stopping with best_acc: ", best_acc)
        #         break

        # writer.add_scalar('Training loss', avg_train_acccuracy, global_step=epoch_idx)
        # writer.add_scalar('Training Accuracy', avg_train_acccuracy, global_step=epoch_idx)
        # writer.add_scalar('Eval Accuracy', avg_test_accuracy, global_step=epoch_idx)
    logging.info('\n')
    logging.info("avg_train_acc: {}".format(avg_train_acccuracy))
    logging.info("avg_val_acc: {}".format(avg_test_accuracy))
    return best_acc, best_epoch

if __name__ == "__main__":
    # writer = SummaryWriter(f'runs//RGNN')
    args = parser.parse_args()
    len_trail = 15 if args.dataset == 'SEED' else 24

    Acc = []
    print('Start time: {}'.format(time.asctime(time.localtime(time.time()))))
    for session in range(3):
        for subject in range(15):
            acc, epoch = main(args, subject+1, session+1)
            Acc.append(acc)
            for idx, acc_value in enumerate(Acc):
                print('Acc for Session--{}, Subject--{} is : {:.4f}'.format(idx//len_trail+1, idx%len_trail+1, acc_value))
            print('Session{}--Subject{}--Acc: {:.3f}--epoch: {}'.format(session+1, subject+1, acc, epoch))
        print('Session{} Accuracy: {:.3f}'.format(session, sum(Acc[-15:]) / 15))
    f_result = np.zeros((15, 2))

    for i in range(15):
        max_two = sorted(Acc[i * 3:3 * (i + 1)])
        f_result[i, 0], f_result[i, 1] = max_two[1], max_two[2]
    result = np.mean(Acc, axis=0)
    final_r = np.mean(f_result)
    standard = np.std(f_result)
    print('result:', result)
    print('final_r:', final_r)
    print('std:', standard)
    print('Session1 Accuracy: {:.3f}'.format(sum(Acc[:15]) / 15))
    print('Session2 Accuracy: {:.3f}'.format(sum(Acc[15:30]) / 15))
    print('Session3 Accuracy: {:.3f}'.format(sum(Acc[30:]) / 15))
    print('Mean Accuracy: {:.3f}'.format(sum(Acc) / len(Acc)))
    print('End time: {}'.format(time.asctime(time.localtime(time.time()))))
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

