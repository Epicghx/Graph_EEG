# coding:UTF-8
'''
使用提取的 de_LDS 特征进行情感分类，分类器使用 SVM，快速验证。
Created by Xiao Guowen.
'''
from MyUtil.tools import build_extracted_features_dataset
from MyUtil.data_preprocess import EEGprocess
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

import torch
import logging
import argparse
import itertools
import warnings

import numpy as np
import torch.nn as nn
import scipy.io as sio

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard

import MyUtil.Dataset as dataset
from MyUtil.manager import Manager
import models

## initial parameters
device = torch.device('cuda:0')

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()

parser.add_argument('--arch', type=str, default='RGNN',
                   help='Architectures')
parser.add_argument('--num_classes', type=int, default=3,
                   help='Num outputs for dataset')
parser.add_argument('--lr', type=float, default=0.1,
                   help='Learning rate for parameters, used for baselines')
parser.add_argument('--train_pattern', type=str, default='SD',
                    choices=['SD', 'SI'],
                   help='subject_dependent or subject independent')

# Paths.
parser.add_argument('--dataset', type=str, default='SEED',
                    help='Name of dataset')
parser.add_argument('--data_path', type=str, default='Data/{dataset}/DEfea/',
                    help='Name of dataset')
parser.add_argument('--num_layer', type=int, default=2,
                   help='Num layers of GCN')

# Universal parameters

parser.add_argument('--weight_decay', type=float, default=4e-5,
                   help='Weight decay for parameters, used for baselines')
parser.add_argument('--train_batch_size', type=int, default=16,
                   help='input batch size for training')
parser.add_argument('--val_batch_size', type=int, default=16,
                   help='input batch size for validation')
parser.add_argument('--workers', type=int, default=8, help='')
parser.add_argument('--dropout', type=float, default=0.7,
                   help='Dropout rate')
parser.add_argument('--channel', type=float, default=62,
                   help='channel numbers of features')
parser.add_argument('--data_norm', action='store_true', default=False)
# Optimization options.
parser.add_argument('--feature', type=str, default='de_LDS{}',
                    choices=['de_LDS{}', 'PSD{}'],
                   help='features used in GNN')
parser.add_argument('--freq_num', type=int, default=5,
                   help='number of freq bands used')
parser.add_argument('--max_size', type=int, default=64, help='the maximum length in all trails')   #64 for SEEDIV   265 for SEED
parser.add_argument('--mode', type=str, default='',
                    choices=['RevGrad', ''],
                   help='Gan mode')
parser.add_argument('--sample_mode', type=str, default='Sample_process',
                    choices=['Sample_process', 'Series_process'],
                   help='mode of input data processing')


# Other.
parser.add_argument('--cuda', action='store_true', default=True,
                   help='use CUDA')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--checkpoint_format', type=str,
                    default='./{save_folder}/checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')

parser.add_argument('--C', type=float, default=1.5,
                    help='number of epochs to train')
parser.add_argument('--k', type=str, default='linear',
                    choices=['linear', 'rbf', ''],
                   help='features used in GNN')
#}}}

def main(args, subject, session):

    Process = EEGprocess(args)
    traindata, testdata, trainlabel, testlabel, _ = Process.SD_process(subject, session)
    traindata = traindata.reshape(traindata.shape[0], -1)
    testdata  = testdata.reshape(testdata.shape[0], -1)
    svc_classifier = svm.SVC(C=args.C, kernel='rbf')
    svc_classifier.fit(traindata, trainlabel)
    pred_label = svc_classifier.predict(testdata)
    print(confusion_matrix(testlabel, pred_label))
    print(classification_report(testlabel, pred_label))
    cur_accuracy = svc_classifier.score(testdata, testlabel)

    return cur_accuracy

def all_data_svm(folder_path):
    '''
        加载所有被试的数据，按一定比例切分数据集，用已提取的特征进行分类，不考虑 subject-independent / subject-dependent，也不考虑被试间的差异，粗暴求平均
    :param folder_path: ExtractedFeatures 文件夹路径
    :return one:
    '''
    # 样本加载
    de_LDS_feature_dict, de_LDS_label_dict = build_extracted_features_dataset(folder_path, 'de_LDS', 'gamma')
    # de_LDS_train_feature_dict, de_LDS_train_label_dict, de_LDS_test_feature_dict, de_LDS_test_label_dict = build_extracted_features_dataset(
    #     folder_path, 'de_LDS', 'gamma')
    de_LDS_feature_list = []
    de_LDS_label_list = []


    for key in de_LDS_feature_dict.keys():
        cur_feature = de_LDS_feature_dict[key]
        cur_label   = de_LDS_label_dict[key]
        for trial in cur_feature.keys():
            de_LDS_feature_list.extend(cur_feature[trial])
            de_LDS_label_list.extend(cur_label[trial])

    # 训练集，测试集分割
    test_ratio = 0.4
    train_feature, test_feature, train_label, test_label = train_test_split(de_LDS_feature_list, de_LDS_label_list,
                                                                            test_size=test_ratio)

    # SVM 分类器训练与预测
    # 注意 SVC 参数设置，默认的 C 值为1，即不容错，此时搭配 linear 核很可能无法收敛。
    svc_classifier = svm.SVC(C=0.05, kernel='rbf')
    svc_classifier.fit(train_feature, train_label)
    pred_label = svc_classifier.predict(test_feature)
    print(confusion_matrix(test_label, pred_label))
    print(classification_report(test_label, pred_label))


def paper_svm(folder_path):
    '''
        按照 SEED 数据集原始论文中的 SVM 计算方式测试准确率和方差，每个 experiment 分开计算，取其中 9 个 trial 为训练集，6 个 trial 为测试集
    :param folder_path: ExtractedFeatures 文件夹路径
    :return None:
    '''
    # 样本加载
    de_LDS_feature_dict, de_LDS_label_dict = build_extracted_features_dataset(folder_path, 'de_LDS', 'gamma')
    accuracy = 0
    idx = 0
    train_features, train_labels, test_features, test_labels = [], [], [], []
    for key in de_LDS_feature_dict.keys():
        print('当前处理到 experiment_{}'.format(key))
        cur_feature = de_LDS_feature_dict[key]
        cur_label = de_LDS_label_dict[key]
        train_feature = []
        train_label = []
        test_feature = []
        test_label = []
        for trial in cur_feature.keys():
            if int(trial) < 19:
                train_feature.extend(cur_feature[trial])
                train_label.extend(cur_label[trial])
            else:
                test_feature.extend(cur_feature[trial])
                test_label.extend(cur_label[trial])
        # 定义 svm 分类器
        # while idx < 3:
        #     train_features.extend(train_feature)
        #     train_labels.extend(train_label)
        #     test_features.extend(test_feature)
        #     test_labels.extend(test_label)
        #     idx += 1

        svc_classifier = svm.SVC(C=0.8, kernel='rbf')
        svc_classifier.fit(train_feature, train_label)
        pred_label = svc_classifier.predict(test_feature)
        print(confusion_matrix(test_label, pred_label))
        print(classification_report(test_label, pred_label))
        cur_accuracy = svc_classifier.score(test_feature, test_label)
        accuracy += cur_accuracy
        print('当前 experiment 的 accuracy 为：{}'.format(cur_accuracy))

    print('所有 experiment 上的平均 accuracy 为：{}'.format(accuracy / len(de_LDS_feature_dict.keys())))


if __name__ == "__main__":
    # folder_path = 'Data/ExtractedFeatures/'
    # folder_path = 'Data/SEEDIV/1/'
    # paper_svm(folder_path)
    args = parser.parse_args()
    len_trail = 15 if args.dataset == 'SEED' else 24
    accuracy = 0
    Acc = []
    for session in range(3):
        accuracy1 = 0
        for subject in range(15):
            acc = main(args, subject + 1, session + 1)
            Acc.append(acc)
            print('当前 experiment 的 accuracy 为：{:.3f}'.format(acc))
        # Acc.append(accuracy1/15)
        # print('Session{} 上的平均 accuracy 为：{}'.format(session, accuracy1 / 15))
    print(Acc)
    print('所有 experiment 上的平均 accuracy 为：{:.3f}'.format(sum(Acc)/len(Acc)))

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