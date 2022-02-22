import os
import numpy as np
import scipy.io as sio


class EEGprocess(object):

    def __init__(self, args):

        # super(EEGprocess, self).__init__()
        self.args = args

        return

    def Series_process(self, session):

        global trails
        dir = self.args.data_path.format(dataset=self.args.dataset) + str(session) + '/'
        file = os.listdir(dir) # data_path means root path of data: 'Data/'
        feature = self.args.feature
        freq_num = self.args.freq_num
        Data = []
        Label = []
        Sample_num = np.zeros(len(file))

        if self.args.dataset == "SEEDIV":
            session1_label = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
            session2_label = [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
            session3_label = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
            trails = 24
            max_size = 64
        elif self.args.dataset == "SEED":
            session1_label = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
            session2_label = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
            session3_label = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
            trails = 15
            max_size = 265
        label = np.array(eval('session{}_label'.format(session)))

        for i in range(len(file)):
            data = sio.loadmat(dir + file[i])
            if i < len(file):  ## 15
                for j in range(trails):
                    de_LDS = data[feature.format(j+1)]
                    pre_pad_size = int(np.floor((max_size - np.size(de_LDS, 1)) /2))
                    post_psd_size = max_size - pre_pad_size - np.size(de_LDS, 1)
                    de_LDS = np.pad(de_LDS, ((0, 0), (pre_pad_size, post_psd_size), (0, 0)), 'constant',
                                    constant_values = 0).reshape(62, max_size * freq_num)
                    de_LDS = np.expand_dims(de_LDS, axis = 0)
                    Data.append(de_LDS)
                    Label.append(label[j])
                    Sample_num[i] = trails

        Data = np.concatenate(Data, axis=0)
        Label = np.array(Label).reshape(-1, 1).astype(int)
        Sample_num = Sample_num.reshape(-1, 1).astype(int)



        return Data, Label, Sample_num

    def Sample_process(self, session):

        global trails
        dir = self.args.data_path.format(dataset=self.args.dataset) + str(session) + '/'
        file = os.listdir(dir)
        feature = self.args.feature
        Data = []
        Label = []
        Sample_Num = np.zeros(len(file))

        if self.args.dataset == "SEEDIV":
            session1_label = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
            session2_label = [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
            session3_label = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
            trails = 24
        elif self.args.dataset == "SEED":
            session1_label = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
            session2_label = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
            session3_label = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
            trails = 15

        label = np.array(eval('session{}_label'.format(session)))

        for i in range(len(file)):
            data = sio.loadmat(dir + file[i])
            if i < len(file):
                for j in range(trails):
                    de_LDS = data[feature.format(j+1)]
                    Data.append(de_LDS)
                    Label.append([label[j]] * np.size(de_LDS, 1))
                    Sample_Num[i] += np.size(de_LDS, 1)

        Data  = np.concatenate(Data,  axis = 1).transpose([1, 0, 2])
        Label = np.concatenate(Label, axis = 0).reshape(-1, 1).astype(int)
        Sample_Num = Sample_Num.reshape(-1, 1).astype(int)


        return Data, Label, Sample_Num

# if  __name__== "__main__":
#
#     session = 1
#     dir = '../Data/' + str(session) + '/'
#     feature = 'de_LDS{}'
#
#     max_size = 64
#     channel_num = 5
#
#     # Data, Label = Series_process(dir, feature, session, max_size, channel_num)
#     # Data, Label, Sample_Num= Sample_process(dir, feature, session, max_size, channel_num)
#     #
#     # np.save( 'Data/Train_data{}'.format(session) + '.npy', Data)
#     # np.save('Data/Train_label{}'.format(session) + '.npy', Label)
