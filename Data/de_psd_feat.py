import os
import numpy as np
import scipy.io as sio
from scipy.fftpack import fft, ifft
import scipy.signal as ss
import math

raw_path = './SEED/Preprocessed_EEG/'
new_path = './SEED/DEfea/'
fs = 200

def DE_PSD(data, sampling_rate, freq_start, freq_end, fs, window_size):
    # data (channel, time_len)
    WindowPoints = fs * window_size

    fStartNum = np.zeros([len(freq_start)], dtype=int)
    fEndNum = np.zeros([len(freq_end)], dtype=int)
    for i in range(0, len(freq_start)):
        fStartNum[i] = int(freq_start[i] / fs * sampling_rate)
        fEndNum[i] = int(freq_end[i] / fs * sampling_rate)

    n = data.shape[0]
    m = data.shape[1]

    # print(m,n,l)
    psd = np.zeros([n, len(freq_start)])
    de = np.zeros([n, len(freq_start)])
    # Hanning window
    Hlength = window_size * fs
    # Hwindow=hanning(Hlength)
    # Hwindow = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength)) for n in range(Hlength)])
    Hwindow = ss.windows.hann(200)

    WindowPoints = fs * window_size
    dataNow = data[0:n]
    for j in range(0, n):
        temp = dataNow[j]
        Hdata = temp * Hwindow
        FFTdata = fft(Hdata, sampling_rate)
        magFFTdata = abs(FFTdata[0:int(sampling_rate / 2)])
        for p in range(0, len(freq_start)):
            E = 0
            # E_log = 0
            for p0 in range(fStartNum[p], fEndNum[p] + 1):
                E = E + magFFTdata[p0] * magFFTdata[p0]
            #    E_log = E_log + log2(magFFTdata(p0)*magFFTdata(p0)+1)
            E = E / (fEndNum[p] - fStartNum[p] + 1)
            psd[j][p] = E
            de[j][p] = math.log(100 * E, 2)
            # de(j,i,p)=log2((1+E)^4)

    return psd, de

for session in range(1, 4):
    skip_set = {'label.mat', 'readme.txt'}
    raw_data_path = raw_path + '{}/'.format(session)
    matfile = os.listdir(raw_data_path)
    # matfile.remove('label.mat')
    for f in matfile:
        data = sio.loadmat(raw_data_path + f)
        fea = []
        for trail in range(1, 16):
            DE_LDS = []
            eegkeys = [k for k in data.keys() if 'eeg' in k]
            eeg = data[eegkeys[trail - 1]][:, :-1]
            for l in range(0, eeg.shape[1], fs):
                _, de = DE_PSD(eeg[:,l : l + fs], fs, [1,4,8,14,31], [3, 7, 13, 30, 50], 200, 1)
                DE_LDS.append(de)
            DE_LDS = np.stack(DE_LDS, 1)
            fea.append(DE_LDS)
        sio.savemat(new_path + '{}/'.format(session) + f, mdict={'de_LDS{}'.format(i+1): fea[i] for i in range(len(fea))} )
        print('Save {}.'.format(f))
    print('Complete session {}'.format(session))