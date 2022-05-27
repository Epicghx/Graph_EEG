# coding:UTF-8
'''
各种处理 SEED Dataset 用到的小函数
Created by Xiao Guowen.
'''
import scipy.io as scio
import numpy as np
import os
# import torch
# import torch.nn.functional as F
# from torch.utils import data
import xlrd
from memory_profiler import profile
import pickle


def get_labels(label_path):
    '''
        得到15个 trials 对应的标签
    :param label_path: 标签文件对应的路径
    :return: list，对应15个 trials 的标签，1 for positive, 0 for neutral, -1 for negative
    '''
    print(os.getcwd())

    return scio.loadmat(label_path, verify_compressed_data_integrity=False)['label'][0]


def label_2_onehot(folder_path, label_list):
    '''
        将原始-1， 0， 1标签转化为独热码形式
    :param label_list: 原始标签列表
    :return label_onehot: 独热码形式标签列表
    '''
    if 'SEEDIV' in folder_path:
        look_up_table = {0: [1, 0, 0, 0],
                         1: [0, 1, 0, 0],
                         2: [0, 0, 1, 0],
                         3: [0, 0, 0, 1]}

    else:
        look_up_table = {-1: [1, 0, 0],
                         0: [0, 1, 0],
                         1: [0, 0, 1]}
    label_onehot = [np.asarray(look_up_table[label]) for label in label_list]
    return label_onehot


def get_frequency_band_idx(frequency_band):
    '''
        获得频带对应的索引，仅对 ExtractedFeatures 目录下的数据有效
    :param frequency_band: 频带名称，'delta', 'theta', 'alpha', 'beta', 'gamma'
    :return idx: 频带对应的索引
    '''
    lookup = {'delta': 0,
              'theta': 1,
              'alpha': 2,
              'beta': 3,
              'gamma': 4}
    return lookup[frequency_band]


def build_extracted_features_dataset(folder_path, feature_name, frequency_band):
    '''
        将 folder_path 文件夹中的 ExtractedFeatures 数据转化为机器学习常用的数据集，区分开不同 trial 的数据
        ToDo: 增加 channel 的选择，而不是使用所有的 channel
    :param folder_path: ExtractedFeatures 文件夹对应的路径
    :param feature_name: 需要使用的特征名，如 'de_LDS'，'asm_LDS' 等，以 de_LDS1 为例，维度为 62 * 235 * 5，235为影片长度235秒，每秒切分为一个样本，62为通道数，5为频带数
    :param frequency_band: 需要选取的频带，'delta', 'theta', 'alpha', 'beta', 'gamma'
    :return feature_vector_dict, label_dict: 分别为样本的特征向量，样本的标签，key 为被试名字，val 为该被试对应的特征向量或标签的 list，方便 subject-independent 的测试
    '''
    frequency_idx = get_frequency_band_idx(frequency_band)
    labels = get_labels(os.path.join(folder_path, 'label.mat'))
    if 'SEEDIV' in folder_path:
        len_trails = 25
    else:
        len_trails = 16
    feature_vector_dict = {}
    label_dict = {}
    try:
        all_mat_file = os.walk(folder_path)
        skip_set = {'label.mat', 'readme.txt'}
        file_cnt = 0
        for path, dir_list, file_list in all_mat_file:
            for file_name in file_list:
                file_cnt += 1
                trails_id = file_name.split('_')[0]
                print('当前已处理到{}，总进度{}/{}'.format(file_name, file_cnt, len(file_list)))
                if file_name not in skip_set:
                    all_features_dict = scio.loadmat(os.path.join(folder_path, file_name),
                                                     verify_compressed_data_integrity=False)
                    subject_name = file_name.split('.')[0]
                    feature_vector_trial_dict = {}
                    label_trial_dict = {}
                    for trials in range(1, len_trails):
                        feature_vector_list = []
                        label_list = []
                        cur_feature = all_features_dict[feature_name + str(trials)]
                        # cur_feature = np.asarray(cur_feature[:, :, frequency_idx]).T  # 转置后，维度为 N * 62, N 为影片长度
                        cur_feature = np.asarray(cur_feature[:, :, :]).transpose([1, 0, 2])
                        feature_vector_list.extend(_.reshape(-1) for _ in cur_feature)
                        # feature_vector_list.extend(_.reshape(-1, 1) for _ in cur_feature)
                        for _ in range(len(cur_feature)):
                            label_list.append(labels[trials - 1])
                        feature_vector_trial_dict[str(trials)] = feature_vector_list
                        label_trial_dict[str(trials)] = label_list
                    feature_vector_dict[subject_name] = feature_vector_trial_dict
                    label_dict[subject_name] = label_trial_dict
                else:
                    continue
    except FileNotFoundError as e:
        print('加载数据时出错: {}'.format(e))

    return feature_vector_dict, label_dict

# def build_extracted_features_dataset(folder_path, feature_name, frequency_band):
#     '''
#         将 folder_path 文件夹中的 ExtractedFeatures 数据转化为机器学习常用的数据集，区分开不同 trial 的数据
#         ToDo: 增加 channel 的选择，而不是使用所有的 channel
#     :param folder_path: ExtractedFeatures 文件夹对应的路径
#     :param feature_name: 需要使用的特征名，如 'de_LDS'，'asm_LDS' 等，以 de_LDS1 为例，维度为 62 * 235 * 5，235为影片长度235秒，每秒切分为一个样本，62为通道数，5为频带数
#     :param frequency_band: 需要选取的频带，'delta', 'theta', 'alpha', 'beta', 'gamma'
#     :return feature_vector_dict, label_dict: 分别为样本的特征向量，样本的标签，key 为被试名字，val 为该被试对应的特征向量或标签的 list，方便 subject-independent 的测试
#     '''
#     frequency_idx = get_frequency_band_idx(frequency_band)
#     labels = get_labels(os.path.join(folder_path, 'label.mat'))
#     train_feature_vector_dict = {}
#     test_feature_vector_dict = {}
#     train_label_dict = {}
#     test_label_dict = {}
#     try:
#         all_mat_file = os.walk(folder_path)
#         skip_set = {'label.mat', 'readme.txt'}
#         file_cnt = 0
#         for path, dir_list, file_list in all_mat_file:
#             for file_name in file_list:
#                 file_cnt += 1
#                 print('当前已处理到{}，总进度{}/{}'.format(file_name, file_cnt, len(file_list)))
#                 if file_name not in skip_set:
#                     all_features_dict = scio.loadmat(os.path.join(folder_path, file_name),
#                                                      verify_compressed_data_integrity=False)
#                     subject_name = file_name.split('.')[0]
#                     train_feature_vector_trial_dict = {}
#                     test_feature_vector_trial_dict  = {}
#                     train_label_trial_dict = {}
#                     test_label_trial_dict  = {}
#                     for trials in range(1, 16):
#                         train_feature_vector_list = []
#                         test_feature_vector_list  = []
#                         train_label_list = []
#                         test_label_list  = []
#                         cur_feature = all_features_dict[feature_name + str(trials)]
#                         cur_feature = np.asarray(cur_feature[:, :, frequency_idx]).T  # 转置后，维度为 N * 62, N 为影片长度
#                         if trials <= 9:
#                             train_feature_vector_list.extend(_ for _ in cur_feature)
#                             for _ in range(len(cur_feature)):
#                                 train_label_list.append(labels[trials - 1])
#                             train_feature_vector_trial_dict[str(trials)] = train_feature_vector_list
#                             train_label_trial_dict[str(trials)] = train_label_list
#                         else:
#                             test_feature_vector_list.extend(_ for _ in cur_feature)
#                             for _ in range(len(cur_feature)):
#                                 test_label_list.append(labels[trials - 1])
#                             test_feature_vector_trial_dict[str(trials)] = train_feature_vector_list
#                             test_label_trial_dict[str(trials)] = test_label_list
#                     train_feature_vector_dict[subject_name] = train_feature_vector_trial_dict
#                     test_feature_vector_dict[subject_name] = train_feature_vector_trial_dict
#                     train_label_dict[subject_name] = test_label_trial_dict
#                     test_label_dict[subject_name] = test_label_trial_dict
#                 else:
#                     continue
#     except FileNotFoundError as e:
#         print('加载数据时出错: {}'.format(e))
#
#     return train_feature_vector_dict, test_feature_vector_dict, train_label_dict, test_label_dict

@profile
def build_preprocessed_eeg_dataset_CNN(folder_path):
    '''
        预处理后的 EEG 数据维度为 62 * N，其中62为 channel 数量， N 为采样点个数（已下采样到200 Hz）
        此函数将预处理后的 EEG 信号转化为 CNN 网络所对应的数据格式，即 62 * 200 的二维输入（每 1s 的信号作为一个样本）,区分开不同 trial 的数据
    :param folder_path: Preprocessed_EEG 文件夹对应的路径
    :return feature_vector_dict, label_dict: 分别为样本的特征向量，样本的标签，key 为被试名字，val 为该被试对应的特征向量或标签的 list，方便 subject-independent 的测试
    '''
    feature_vector_dict = {}
    label_dict = {}
    labels = get_labels(os.path.join(folder_path, 'label.mat'))
    try:
        all_mat_file = os.walk(folder_path)
        skip_set = {'label.mat', 'readme.txt'}
        file_cnt = 0
        for path, dir_list, file_list in all_mat_file:
            for file_name in file_list:
                file_cnt += 1
                print('当前已处理到{}，总进度{}/{}'.format(file_name, file_cnt, len(file_list)))
                if file_name not in skip_set:
                    all_trials_dict = scio.loadmat(os.path.join(folder_path, file_name),
                                                   verify_compressed_data_integrity=False)
                    experiment_name = file_name.split('.')[0]
                    feature_vector_trial_dict = {}
                    label_trial_dict = {}
                    for key in all_trials_dict.keys():
                        if 'eeg' not in key:
                            continue
                        feature_vector_list = []
                        label_list = []
                        cur_trial = all_trials_dict[key]  # 维度为 62 * N，每200个采样点截取一个样本，不足200时舍弃
                        length = len(cur_trial[0])
                        pos = 0
                        while pos + 200 <= length:
                            feature_vector_list.append(np.asarray(cur_trial[:, pos:pos + 200], dtype=np.float32))
                            raw_label = labels[int(key.split('_')[-1][3:]) - 1]  # 截取片段对应的 label，-1, 0, 1
                            label_list.append(raw_label)
                            pos += 200
                        trial = key.split('_')[1][3:]
                        feature_vector_trial_dict[trial] = np.asarray(feature_vector_list, dtype=np.float32)
                        label_trial_dict[trial] = np.asarray(label_2_onehot(folder_path, label_list))

                    feature_vector_dict[experiment_name] = feature_vector_trial_dict
                    label_dict[experiment_name] = label_trial_dict
                else:
                    continue

    except FileNotFoundError as e:
        print('加载数据时出错: {}'.format(e))

    return feature_vector_dict, label_dict


def get_channel_order(file_path):
    '''
        获取 SEED 数据集采集时各个通道对应的索引
    :param file_path: channel-order.xlsx 文件路径
    :return channel_order_dict: 索引-通道字典，eg. 0: 'FP1'
    '''
    try:
        channel_file = xlrd.open_workbook(file_path)
        table = channel_file.sheet_by_index(0)
        rows = table.nrows
        channel_order_dict = {}
        for i in range(rows):
            channel_order_dict[i] = table.cell_value(i, 0)
    except FileNotFoundError:
        print('Error: 未找到指定文件！')
    return channel_order_dict


def get_desire_channel_order():
    '''
        按电极的空间排布方式重新整理数据，抽象为一个 8 * 9 的二维平面，行之间按 Z 中间对齐，第 0, 6, 7 行需要补充全 0 数据
    :return channel_order_desire_dict: 通道-索引字典，eg: 'AF3': 2
    '''
    channel_order_desire_dict = {'AF3': 2, 'FP1': 3, 'FPZ': 4, 'FP2': 5, 'AF4': 6,
                                 'F7': 9, 'F5': 10, 'F3': 11, 'F1': 12, 'FZ': 13, 'F2': 14, 'F4': 15, 'F6': 16, 'F8': 17,
                                 'FT7': 18, 'FC5': 19, 'FC3': 20, 'FC1': 21, 'FCZ': 22, 'FC2': 23, 'FC4': 24, 'FC6': 25, 'FT8': 26,
                                 'T7': 27, 'C5': 28, 'C3': 29, 'C1': 30, 'CZ': 31, 'C2': 32, 'C4': 33, 'C6': 34, 'T8': 35,
                                 'TP7': 36, 'CP5': 37, 'CP3': 38, 'CP1': 39, 'CPZ': 40, 'CP2': 41, 'CP4': 42, 'CP6': 43, 'TP8': 44,
                                 'P7': 45, 'P5': 46, 'P3': 47, 'P1': 48, 'PZ': 49, 'P2': 50, 'P4': 51, 'P6': 52, 'P8': 53,
                                 'PO7': 55, 'PO5': 56, 'PO3': 57, 'POZ': 58, 'PO4': 59, 'PO6': 60, 'PO8': 61,
                                 'CB1': 65, 'O1': 66, 'OZ': 67, 'O2': 68, 'CB2': 69}
    return channel_order_desire_dict


@profile
def generate_preprocessed_eeg_dataset_CNN_3D(data_folder_path, channel_file_path, save_path):
    '''
        将 preprocessed_eeg 数据从 类似 62 * 200 的维度调整为 200 * 8 * 9 维度，并保存
    :param data_folder_path: Preprocesse_EEG 文件夹路径
    :param channel_file_path: channel-order.xlsx 文件路径
    :param save_path: 本地的保存路径
    '''
    # 获取 channel 的索引信息，获取维度为 62 * 200 的 eeg 数据
    raw_feature_vector_dict, label_dict = build_preprocessed_eeg_dataset_CNN(data_folder_path)
    channel_order_dict = get_channel_order(channel_file_path)
    channel_order_desire_dict = get_desire_channel_order()
    cnt = 1
    # 整理为 200 * 8 * 9 的维度
    for experiment in raw_feature_vector_dict.keys():
        print('当前处理到 {}，进度{}/{}'.format(experiment, cnt, len(raw_feature_vector_dict.keys())))
        cnt += 1
        experiment_dict = {}
        for trial in raw_feature_vector_dict[experiment].keys():
            raw_trial_data = raw_feature_vector_dict[experiment][trial]
            trial_data = []
            for _, raw_sample in enumerate(raw_trial_data):
                sample = np.zeros((8, 9, 200), dtype=np.float32)
                for raw_channel_index in range(62):
                    channel_index = channel_order_desire_dict[channel_order_dict[raw_channel_index]]
                    sample[channel_index // 9][channel_index % 9] = raw_sample[raw_channel_index]
                swap_sample = sample.transpose((2, 0, 1)).copy()
                trial_data.append(swap_sample)
            experiment_dict[trial] = trial_data
        raw_feature_vector_dict[experiment] = None
        experiment_path = os.path.join(save_path, experiment + '.pickle')
        with open(experiment_path, 'wb') as experiment_file:
            pickle.dump(experiment_dict, experiment_file)
    label_path = os.path.join(save_path, 'label.pickle')
    with open(label_path, 'wb') as label_file:
        pickle.dump(label_dict, label_file)


def build_preprocessed_eeg_dataset_CNN_3D(file_path):
    '''
        加载本地已处理好的 3D 数据
    :param file_path: 文件夹路径
    :return feature_vector_dict, label_dict: 分别为样本的特征向量，样本的标签，key 为被试名字，val 为该被试对应的特征向量或标签的 list，方便 subject-independent 的测试
    '''
    all_file = os.listdir(file_path)
    feature_vector_dict = {}
    for file_name in all_file:
        with open(os.path.join(file_path, file_name), 'rb') as file:
            key = file_name.split('.')[0]
            if key == 'readme':
                continue
            elif key == 'label':
                label_dict = pickle.load(file)
            else:
                feature_vector_dict[key] = pickle.load(file)
    return feature_vector_dict, label_dict


def subject_independent_data_split(feature_vector_dict, label_dict, test_subject_set):
    '''
        使用 subject_independent 的方式做数据切分
    :param feature_vector_dict: build_preprocessed_eeg_dataset_CNN 函数返回的 feature_vector_dict
    :param label_dict: build_preprocessed_eeg_dataset_CNN 函数返回的 label_dict
    :param test_subject_set: 留一法，用作测试集的 subject
    :return train_feature, train_label, test_feature, test_label: 训练特征，训练标签，测试特征，测试标签
    '''
    train_feature = []
    train_label = []
    test_feature = []
    test_label = []
    for experiment in feature_vector_dict.keys():
        subject = experiment.split('_')[0]
        for trial in feature_vector_dict[experiment].keys():
            if subject in test_subject_set:
                test_feature.extend(feature_vector_dict[experiment][trial])
                test_label.extend(label_dict[experiment][trial])
            else:
                train_feature.extend(feature_vector_dict[experiment][trial])
                train_label.extend(label_dict[experiment][trial])
    return train_feature, train_label, test_feature, test_label


# class RawEEGDataset(torch.utils.data.Dataset):
#     def __init__(self, feature_list, label_list, desire_shape, norm_dim):
#         self.feature_list = feature_list
#         self.label_list = label_list
#         self.desire_shape = desire_shape
#         self.norm_dim = norm_dim
#
#     def __getitem__(self, index):
#         self.feature_list[index] = self.feature_list[index].reshape(self.desire_shape)
#         # N * M * ... * 200 * X * ... * Y，对 200 这个维度进行归一化
#         feature = F.normalize(torch.from_numpy(self.feature_list[index]).float(), p=2, dim=self.norm_dim)
#         label = torch.from_numpy(self.label_list[index]).long()
#         label = torch.argmax(label)
#         return feature, label
#
#     def __len__(self):
#         return len(self.label_list)


# fe, la = build_preprocessed_eeg_dataset_CNN_3D('../../local_data_3D/')
