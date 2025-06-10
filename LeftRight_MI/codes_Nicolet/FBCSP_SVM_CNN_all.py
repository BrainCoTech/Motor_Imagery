import numpy as np
import mne
import joblib
from sklearn import svm
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 或者试试 'TkAgg'
import seaborn as sns
from scipy import signal

import tensorflow as tf
from keras import utils as np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Activation, Permute, Dropout
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import SeparableConv2D, DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.layers import SpatialDropout2D
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten
from keras.constraints import max_norm
from keras import backend as K


def emg_plot(data):
    y_interval = np.max(np.abs(data))
    ch_num = data.shape[0]
    # # with trigger
    # for i in range(ch_num):
    #     if i < ch_num-1:
    #         plt.plot(data[i, :]+y_interval*(ch_num-i))
    #     else:
    #         # trigger
    #         plt.plot(data[i, :]*y_interval*0.1 + y_interval * (ch_num - i))

    # # without trigger
    for i in range(ch_num):
        plt.plot(data[i, :]+y_interval*(ch_num-i))

    plt.ylim((0, (ch_num+1)*y_interval))
    # plt.xlabel('time')
    # plt.ylabel('channels (interval/'+str(int(y_interval))+r'$\mu$V'+')')
    # plt.yticks([y_interval*(ch_num-0), y_interval*(ch_num-1), y_interval*(ch_num-2),
    #             y_interval*(ch_num-3), y_interval*(ch_num-4), y_interval*(ch_num-5),
    #             y_interval*(ch_num-6), y_interval*(ch_num-7), y_interval*(ch_num-8)],
    #            ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'trigger'])
    # plt.show()


def emg_plot_fft(f, data):
    y_interval = np.max(np.abs(data))
    ch_num = data.shape[0]
    for i in range(ch_num):
        plt.plot(f, data[i, :]+y_interval*(ch_num-i))

    plt.ylim((0, (ch_num+1)*y_interval))


def notch_filter(data, f0, fs):
    Q = 30  # Quality factor
    # Design notch filter
    b, a = signal.iirnotch(f0, Q, fs)
    # freq, h = signal.freqz(b, a, fs=fs)
    # plt.plot(freq, 20*np.log10(abs(h)))
    # plt.show()
    # filtering
    y = signal.filtfilt(b, a, data)

    # plt.subplot(2,2,1)
    # emg_plot(data)
    # plt.subplot(2, 2, 2)
    # N = data.shape[1]  # 样本点个数
    # X = np.fft.fft(data)
    # X_mag = np.abs(X) / N  # 幅值除以N倍
    # f_plot = np.arange(int(N / 2 + 1))  # 取一半区间
    # X_mag_plot = 2 * X_mag[:, 0:int(N / 2 + 1)]  # 取一半区间
    # X_mag_plot[:, 0] = X_mag_plot[:, 0] / 2  # Note: DC component does not need to multiply by 2
    # emg_plot_fft(fs*f_plot/N, X_mag_plot)
    # plt.subplot(2,2,3)
    # emg_plot(y)
    # plt.subplot(2, 2, 4)
    # X_ = np.fft.fft(y)
    # X_mag_ = np.abs(X_) / N
    # X_mag_plot_ = 2 * X_mag_[:, 0:int(N / 2 + 1)]  # 取一半区间
    # X_mag_plot_[:, 0] = X_mag_plot_[:, 0] / 2  # Note: DC component does not need to multiply by 2
    # emg_plot_fft(fs*f_plot/N, X_mag_plot_)
    # plt.show()

    return y


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = fs * 0.5  # 奈奎斯特采样频率
    low, high = lowcut / nyq, highcut / nyq
    # # 滤波器的分子（b）和分母（a）多项式系数向量
    # [b, a] = signal.butter(order, [low, high], analog=False, btype='band', output='ba')
    # # # plot frequency response
    # # w, h = signal.freqz(b, a, worN=2000)
    # # plt.plot(w, abs(h), label="order = %d" % order)
    # y = signal.filtfilt(b, a, data)

    # # # # 滤波器的二阶截面表示
    sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
    # # # plot frequency response
    # w, h = signal.freqz(sos, worN=2000)
    # plt.plot(w, abs(h), label="order = %d" % order)
    # filtering
    y = signal.sosfiltfilt(sos, data)

    # plt.subplot(2, 2, 1)
    # emg_plot(data)
    # plt.subplot(2, 2, 2)
    # N = data.shape[1]  # 样本点个数
    # X = np.fft.fft(data)
    # X_mag = np.abs(X) / N  # 幅值除以N倍
    # f_plot = np.arange(int(N / 2 + 1))  # 取一半区间
    # X_mag_plot = 2 * X_mag[:, 0:int(N / 2 + 1)]  # 取一半区间
    # X_mag_plot[:, 0] = X_mag_plot[:, 0] / 2  # Note: DC component does not need to multiply by 2
    # emg_plot_fft(fs * f_plot / N, X_mag_plot)
    # plt.subplot(2, 2, 3)
    # emg_plot(y)
    # plt.subplot(2, 2, 4)
    # X_ = np.fft.fft(y)
    # X_mag_ = np.abs(X_) / N
    # X_mag_plot_ = 2 * X_mag_[:, 0:int(N / 2 + 1)]  # 取一半区间
    # X_mag_plot_[:, 0] = X_mag_plot_[:, 0] / 2  # Note: DC component does not need to multiply by 2
    # emg_plot_fft(fs * f_plot / N, X_mag_plot_)
    # plt.show()

    return y


def read_data_edf(file_name):
    raw = mne.io.read_raw_edf(file_name, preload=True)
    # Clean channel names to be able to use a standard 1005 montage
    new_names = {'EEG Fp1-REF': 'FC5', 'EEG Fp2-REF': 'FC3', 'EEG F7-REF': 'FC1', 'EEG F3-REF': 'FCz',
                 'EEG Fz-REF': 'FC2', 'EEG F4-REF': 'FC4', 'EEG F8-REF': 'FC6',
                 'EEG A1-REF': 'C5', 'EEG T3-REF': 'C3', 'EEG C3-REF': 'C1', 'EEG Cz-REF': 'Cz',
                 'EEG C4-REF': 'C2', 'EEG T4-REF': 'C4', 'EEG A2-REF': 'C6',
                 'EEG T5-REF': 'CP5', 'EEG P3-REF': 'CP3', 'EEG Pz-REF': 'CP1', 'EEG P4-REF': 'CPz',
                 'EEG T6-REF': 'CP2', 'EEG O1-REF': 'CP4', 'EEG O2-REF': 'CP6'}

    raw.rename_channels(new_names)
    raw.info['bads'] = ['EEG ROC-REF', 'EEG LOC-REF', 'ECG EKG-REF', 'Photic-REF', 'IBI', 'Bursts', 'Suppr']
    # raw.info['bads'] = ['FC5', 'FC3', 'FCz', 'FC4', 'FC1', 'FC6', 'FC2',
    #                     'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'Cz',
    #                     'CP3', 'CPz', 'CP4', 'CP6', 'CP5', 'CP2', 'CP1',
    #                     'EEG ROC-REF', 'EEG LOC-REF', 'ECG EKG-REF', 'Photic-REF', 'IBI', 'Bursts', 'Suppr']

    # raw.set_eeg_reference(ref_channels=['EEG ROC-REF', 'EEG LOC-REF'])
    # raw.apply_proj()

    ch_names = list(set(raw.ch_names) - set(raw.info['bads']))  # 得到需要的channels(set会重排顺序)
    ch_names.sort(key=raw.ch_names.index)  # 排序为raw.ch_names的导联顺序

    # raw.plot(block=True, scalings=1e-4)

    # # raw filtering
    freq_win = (4, 20)
    raw.filter(freq_win[0], freq_win[1])

    events_from_annot, event_dict = mne.events_from_annotations(raw)
    print(event_dict)
    event_dicts = {'LeftMI': 3, 'RightMI': 4}

    t_min = -1
    t_max = 4

    epochs = mne.Epochs(raw, events_from_annot, event_dicts, picks=ch_names, reject_by_annotation=True,
                        tmin=t_min, tmax=t_max, baseline=(t_min, 0), preload=True, reject=dict(eeg=1e-4))  # , reject=dict(eeg=1e-4) baseline=(None, None)
    # epochs.plot(block=True, n_epochs=10)

    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    epochs.set_montage(ten_twenty_montage)
    # epochs['LeftMI'].compute_psd().plot()
    # epochs['RightMI'].compute_psd().plot()

    ##############################################################################
    # # Time-frequency analysis
    # # ------------------------------------
    # # freqs = np.logspace(*np.log10([freq_win[0], freq_win[1]]), num=int((freq_win[1]-freq_win[0])/2))  # define frequencies of interest (log-spaced)
    #
    # for i in range(8):
    #     freqs = np.linspace(freq_win[0], freq_win[1], num=int((freq_win[1]-freq_win[0])/2))  # define frequencies of interest (linear-spaced)
    #     n_cycles = freqs / 2.  # 7 freqs / 2.  # different number of cycle per frequency
    #     baseline = [-1, 0]
    #     power, itc = mne.time_frequency.tfr_morlet(epochs['LeftMI'][i*5:(i+1)*5], freqs=freqs, n_cycles=n_cycles, use_fft=True,
    #                                                return_itc=True, decim=3, n_jobs=1)
    #     power.plot_topo(baseline=baseline, tmin=t_min + 0.2, tmax=t_max - 0.2, mode='logratio', title='Average power')
    #     # power.plot([25], baseline=(-1, 0), mode='logratio', title=power.ch_names[25])
    #     power, itc = mne.time_frequency.tfr_morlet(epochs['RightMI'][i*5:(i+1)*5], freqs=freqs, n_cycles=n_cycles, use_fft=True,
    #                                                return_itc=True, decim=3, n_jobs=1)
    #     power.plot_topo(baseline=baseline, tmin=t_min + 0.2, tmax=t_max - 0.2, mode='logratio', title='Average power')

    X = epochs.get_data()[:, :, int((-t_min+0.5)*500):int((-t_min+0.5+2)*500)]
    y = epochs.events[:, 2] - 3

    # # 得到频域信号
    # X_fft = np.zeros((X.shape[0], X.shape[1], 24))  # 24个fft点（8-20Hz）
    # for i in range(X.shape[0]):
    #     N = X[i].shape[1]  # 样本点个数
    #     X_ = np.fft.fft(X[i])
    #     X_mag = np.abs(X_) / N  # 幅值除以N倍
    #     f_plot = np.arange(int(N / 2 + 1))  # 取一半区间
    #     X_mag_plot = 2 * X_mag[:, 0:int(N / 2 + 1)]  # 取一半区间
    #     X_mag_plot[:, 0] = X_mag_plot[:, 0] / 2  # Note: DC component does not need to multiply by 2
    #     # emg_plot_fft(500*f_plot/N, X_mag_plot)
    #     X_fft[i] = X_mag_plot[:, 16:40]
    # X = X_fft.copy()

    return X, y


def read_data_txt(file_name):
    raw_data = []
    with open(file_name, 'r') as f:
        data_lists = f.readlines()
        for data_list in data_lists:
            data1 = data_list.strip('\n')  # 去掉开头和结尾的换行符
            data1 = data1.split(',')  # 把','作为间隔符
            data1 = list(map(float, data1))  # 将list中的string转换为float
            raw_data.append(data1)
        raw_data = np.array(raw_data).T

    data = raw_data[:-1].copy()
    label_row = raw_data[-1].copy()

    # # 信号预处理
    fs = 1000  # Sample frequency (Hz)
    # 1、带通滤波
    f_low = 4  # low frequency
    f_high = 40  # high frequency
    data = butter_bandpass_filter(data, f_low, f_high, fs)
    # 2、循环陷波滤波 notch (band-stop) filters were applied at 50 Hz and
    # its harmonics below the sampling rate to remove the power line noise
    for f0 in np.arange(np.ceil(f_low / 50) * 50, f_high + 1, 50):
        # f0: Frequency to be removed from signal (Hz)
        data = notch_filter(data, int(f0), fs)
    # 3、降采样
    fs = 500
    data = data[:, ::int(1000 / fs)]
    label_row = label_row[::int(1000 / fs)]

    # ************************************************** 画滤波后原始波形图 **************************************************
    # # 单图
    # emg_plot(data)
    # plt.show()
    # # 子图
    # fig, ax = plt.subplots(len(data))
    # y_interval = np.ceil(np.max(np.abs(data)) / 100) * 100
    # for ij in range(len(data)):
    #     ax[ij].plot(data[ij, :])
    #     ax[ij].set_ylim(-y_interval, y_interval)
    # plt.show()

    # 按label大致分割数据
    epoch_data, labels = [], []
    index = 0
    flag = 0
    epoch_length = 5 * fs  # 每个epoch的数据长度（4s）
    # 通过label判断epoch的起始时刻
    while index < len(label_row):
        # label=0是休息，label=1表示左手MI，label=2表示右手MI
        if label_row[index] != 0:
            # 剔除信号幅值过大的试次
            if np.any(abs(data[:, index:index + epoch_length]) > 100):
                pass
            else:
                epoch_data.append(data[:, index:index + epoch_length])
                labels.append(label_row[index])
                # ******以下代码画图用******
                if flag == 0:
                    epoch_data_ = data[:, index:index + epoch_length].copy()
                    flag = 1
                else:
                    epoch_data_ = np.concatenate((epoch_data_, data[:, index:index + epoch_length].copy()), axis=1)
                # ******以上代码画图用******
            # index往后移epoch_length个点
            index += epoch_length
        index += 1

    X = np.array(epoch_data)[:, :, int(0.5*fs):int(3.5*fs)]
    y = np.array(labels).astype(int) - 1

    # ************************************************ 画粗略分段波形图 ************************************************
    # # 单图
    # emg_plot(epoch_data_)
    # for i in range(len(epoch_data)):
    #     plt.axvline(epoch_length * i, c='k')
    # # 子图
    # fig, ax = plt.subplots(epoch_data_.shape[0])
    # y_epoch_length = np.ceil(np.max(np.abs(epoch_data_))/100)*100
    # for ij in range(epoch_data_.shape[0]):
    #     ax[ij].plot(epoch_data_[ij, :])
    #     ax[ij].set_ylim(-y_epoch_length, y_epoch_length)
    #     # ax[ij].set_xticks(range(int(epoch_length/2), epoch_data_.shape[1], epoch_length), range(0, epoch_data.shape[0]))
    #     for i in range(len(epoch_data)):
    #         ax[ij].axvline(epoch_length*i, c='k')
    # plt.tight_layout()
    # plt.show()

    return X, y


def csp_feature(csp_projected_tr, csp_projected_te):
    # extracting the CSP features from each trial
    csp_train_feature = np.zeros((csp_projected_tr.shape[0], csp_projected_tr.shape[1]))
    csp_test_feature = np.zeros((csp_projected_te.shape[0], csp_projected_te.shape[1]))
    csp_proj_tr_te = [csp_projected_tr, csp_projected_te]
    for flag, x in enumerate(csp_proj_tr_te):
        for t, xx in enumerate(x):
            # generating the features as the log variance of the projected signals
            variances_x = np.var(xx, axis=1)
            for f in range(variances_x.shape[0]):
                if flag == 0:
                    csp_train_feature[t, f] = np.log(variances_x[f] / variances_x.sum())
                else:
                    csp_test_feature[t, f] = np.log(variances_x[f] / variances_x.sum())

    return csp_train_feature, csp_test_feature


def csp_projected(x_train, x_test, y_train):
    # 初始化
    # x_train：trials*channels*sampling_points
    trials_train = x_train.shape[0]
    trials_test = x_test.shape[0]
    channels = x_train.shape[1]
    class_labels = np.unique(y_train)  # 不同的分类
    classes = class_labels.shape[0]
    cov_matrix = []
    # 计算每个trial的归一化协方差矩阵
    trial_cov = np.zeros((channels, channels, trials_train))
    for i in range(trials_train):
        E = x_train[i, :, :]
        EE = np.dot(E, E.T)
        trial_cov[:, :, i] = EE/np.trace(EE)  # 计算协方差矩阵
    # 计算每一类样本数据的空间协方差之和, 求各自标签的均值空间协方差矩阵
    for i in range(classes):
        cov_matrix.append(trial_cov[:, :, y_train == class_labels[i]].mean(axis=2))
    # 计算两类数据的空间协方差之和
    cov_total = cov_matrix[0] + cov_matrix[1]
    # 计算特征向量uc和特征值矩阵dt
    uc = np.linalg.eig(cov_total)[1]  # U
    dt = np.linalg.eig(cov_total)[0]  # lamda
    # 验证特征分解
    # dt_diag = np.diag(dt)
    # rr = np.dot(uc, dt_diag)
    # rr = np.dot(rr, uc.T)
    # 特征值要降序排列
    eg_index = np.argsort(-dt)  # argsort()函数返回的是数组值从小到大的"索引值"
    eigenvalues = dt[eg_index]  # 降序排列
    # eigenvalues = sorted(dt, reverse=True)  # 降序排列
    ut = uc[:, eg_index]
    # 白化矩阵
    p = np.dot(np.diag(np.sqrt(1/eigenvalues)), ut.T)
    # 矩阵P作用求公共特征向量transformed_cov1
    transformed_cov1 = np.dot(np.dot(p, cov_matrix[1]), p.T)
    # 计算公共特征向量transformed_cov1的特征向量和特征矩阵
    u1 = np.linalg.eig(transformed_cov1)[1].real
    d1 = np.linalg.eig(transformed_cov1)[0].real
    # d11 = np.linalg.eig(transformed_cov1)[0]#.real
    eg_index1 = np.argsort(-d1)  # argsort()函数返回的是数组值从小到大的"索引值"
    # eigenvalues1 = d1[eg_index1]  # 降序排列
    u1 = u1[:, eg_index1]
    # 以下语句用以验证lamda1+lamda2=1
    # transformed_cov0 = np.dot(np.dot(p, cov_matrix[0]), p.T)
    # d0 = np.linalg.eig(transformed_cov0)[0].real
    # d00 = np.linalg.eig(transformed_cov0)[0]#.real
    # dt01 = np.diag(d0) + np.diag(d1)
    # 计算投影矩阵W
    csp_matrix = np.dot(u1.T, p)
    # 计算特征矩阵
    filter_pairs = 2  # CSP特征选择参数m,CSP特征为2*m个
    Filter = np.vstack((csp_matrix[0:filter_pairs, :], csp_matrix[-filter_pairs:, :]))

    # get the CSP projected signals
    csp_projected_tr = np.zeros((trials_train, 2*filter_pairs, x_train.shape[2]))
    csp_projected_te = np.zeros((trials_test, 2*filter_pairs, x_test.shape[2]))
    x_train_test = [x_train, x_test]
    for flag, x in enumerate(x_train_test):
        for t, xx in enumerate(x):
            # projecting the data onto the CSP filters
            if flag == 0:
                csp_projected_tr[t] = np.dot(Filter, xx)
            else:
                csp_projected_te[t] = np.dot(Filter, xx)

    return csp_projected_tr, csp_projected_te


def FBCSP(X_tr, X_te, Y_tr):
    tr_trials = X_tr.shape[0]  # 训练数据包含的试次数量
    te_trials = X_te.shape[0]  # 测试数据包含的试次数量
    points = X_tr.shape[2]  # 每个trial的点数

    # FBCSP
    fs = 500
    filt = 1  # 滤波器组的个数（8Hz-30Hz，间隔4Hz，重叠2Hz）
    features = 4  # 两类数据，每类数据2个特征，所以一个频带4个特征

    csp_proj_tr = np.zeros((tr_trials, filt*features, points))
    csp_proj_te = np.zeros((te_trials, filt*features, points))

    # 用滤波器组对epoch数据进行滤波，得到epochs_filt_tr和epochs_filt_te
    for freq in range(filt):
        f_low = 8 + freq * 2  # low frequency
        f_high = 12 + freq * 2  # high frequency
        epochs_filt_tr = butter_bandpass_filter(X_tr, f_low, f_high, fs)
        epochs_filt_te = butter_bandpass_filter(X_te, f_low, f_high, fs)

        csp_proj_tr[:, freq*features:(freq+1)*features], csp_proj_te[:, freq*features:(freq+1)*features] =\
            csp_projected(epochs_filt_tr, epochs_filt_te, Y_tr)

    return csp_proj_tr, csp_proj_te


def SVM_classify(X_tr, X_te, Y_tr, Y_te):
    csp_tr, csp_te = csp_feature(X_tr, X_te)

    # # 初始化selector（多种score_func可选，此处列举两种）
    # k_best = 4  # features * filt
    # # selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
    # selector = SelectKBest(score_func=f_classif, k=k_best)
    # # 训练selector
    # selector.fit(csp_tr, Y_tr)
    # # 得到训练集和测试集的最优特征
    # csp_tr = selector.transform(csp_tr)
    # csp_te = selector.transform(csp_te)

    # 初始化SVM分类器
    svm = SVC()
    # 分类器训练
    svm.fit(csp_tr, Y_tr)
    # 通过训练好的SVM得到预测值
    y_pred = svm.predict(csp_te)

    # 将预测值与标签进行对比计算出分数
    acc = accuracy_score(Y_te, y_pred)
    print('识别准确率:', acc)

    # 画混淆矩阵
    CM = confusion_matrix(Y_te, y_pred, labels=[0, 1])
    print('混淆矩阵:\n', CM)  # 打印出来看看
    print('左手MI:', CM[0, 0] / np.sum(CM[0]), ' 右手MI:', CM[1, 1] / np.sum(CM[1]))
    # CM_acc = np.zeros((2, 2))
    # for cla in range(2):
    #     CM_acc[cla] = CM[cla] / np.sum(CM[cla])
    # sns.set()
    # f, ax = plt.subplots()
    # sns.heatmap(CM_acc, annot=True, ax=ax)  # 画热力图
    #
    # ax.set_title('confusion matrix')  # 标题
    # ax.set_xlabel('predict')  # x轴
    # ax.set_ylabel('true')  # y轴
    # plt.show()

    return acc, CM


def EEGNet(nb_classes, Chans=64, Samples=128,
           dropoutRate=0.5, kernLength=64, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:

        1. Depthwise Convolutions to learn spatial filters within a
        temporal convolution. The use of the depth_multiplier option maps
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn
        spatial filters within each filter in a filter-bank. This also limits
        the number of free parameters to fit when compared to a fully-connected
        convolution.

        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions.


    While the original paper used Dropout, we found that SpatialDropout2D
    sometimes produced slightly better results for classification of ERP
    signals. However, SpatialDropout2D significantly reduced performance
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.

    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the
    kernel lengths for double the sampling rate, etc). Note that we haven't
    tested the model performance with this rule so this may not work well.

    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.
    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D.
    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 1))

    ##################################################################
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D, depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


# need these for ShallowConvNet
def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


def ShallowConvNet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.

    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25

    Note that this implementation has not been verified by the original
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations.
    """

    # start the model
    input_main = Input((Chans, Samples, 1))
    block1 = Conv2D(40, (1, 13),
                    input_shape=(Chans, Samples, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(40, (Chans, 1), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 3), strides=(1, 2))(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


def FullyConnected(nb_classes, Chans=5, Samples=1000, dropoutRate=0.5):
    input = Input(shape=(Chans, Samples, 1))
    flatten = Flatten()(input)
    dense = Dense(100)(flatten)
    # dense = Activation('relu')(dense)
    dense = Dense(50)(dense)
    # dense = Activation('relu')(dense)
    # dense = Dense(32)(dense)
    # dense = Activation('relu')(dense)
    dense = Dense(nb_classes)(dense)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input, outputs=softmax)


def cnn_best(nb_classes, Chans=5, Samples=40, dropoutRate=0.5, kernLength=2, C1=16, C2=16, D1=256, norm_rate=0.25):
    input = Input(shape=(Chans, Samples, 1))
    block = Conv2D(C1, (kernLength, kernLength), input_shape=(Chans, Samples, 1))(input)
    block = BatchNormalization()(block)
    block = Activation('relu')(block)
    block = AveragePooling2D((1, 2))(block)
    block = Dropout(dropoutRate)(block)

    dense = Flatten()(block)
    dense = Dense(D1)(dense)
    dense = Dense(nb_classes)(dense)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input, outputs=softmax)


def CNN_classify(X_tr, X_te, Y_tr, Y_te):
    X_train, X_test, Y_train, Y_test = X_tr, X_te, Y_tr, Y_te
    y_te = Y_te.copy()
    X_train, X_validate, Y_train, Y_validate = train_test_split(X_train, Y_train, test_size=0.25, random_state=0, stratify=Y_train)

    kernels, chans, samples = 1, X_train.shape[1], X_train.shape[2]

    # convert labels to one-hot encodings.
    Y_train = np_utils.to_categorical(Y_train)
    Y_validate = np_utils.to_categorical(Y_validate)
    Y_test = np_utils.to_categorical(Y_test)

    # convert data to NHWC (trials, channels, samples, kernels) format.
    X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
    X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
    X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other
    # model configurations may do better, but this is a good starting point)

    # kernLength, F1, D, F2 = 32, 8, 2, 16
    # model = EEGNet(nb_classes=2, Chans=chans, Samples=samples,
    #                dropoutRate=0.5, kernLength=kernLength, F1=F1, D=D, F2=F2,
    #                dropoutType='Dropout')

    # model = ShallowConvNet(nb_classes=2, Chans=chans, Samples=samples, dropoutRate=0.5)

    kernLength, C1, D1 = 7, 8, 256
    model = cnn_best(nb_classes=2, Chans=X_train.shape[1], Samples=X_train.shape[2],
                     dropoutRate=0.5, kernLength=kernLength, C1=C1, C2=32, D1=D1)

    print(model.summary())

    # compile the model and set the optimizers
    opt = Adam()  # learning_rate=1e-5
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # count number of parameters in the model
    numParams = model.count_params()

    model_path = './model/'

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath=model_path + 'checkpoint.h5', verbose=1,
                                   save_best_only=True)

    # callback = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    history = model.fit(X_train, Y_train, batch_size=int(X_train.shape[0] * 0.2), epochs=500,
                        verbose=2, validation_data=(X_validate, Y_validate),
                        callbacks=[checkpointer])  # , class_weight=class_weights
    model.save(model_path + 'model.h5')

    # # 画图
    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    #
    # epochs = range(len(acc))
    #
    # plt.plot(epochs, acc, 'b', label='Training acc')
    # plt.plot(epochs, val_acc, 'r', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.legend()
    #
    # plt.figure()
    #
    # plt.plot(epochs, loss, 'b', label='Training loss')
    # plt.plot(epochs, val_loss, 'r', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()
    #
    # plt.show()

    # testing
    # model1 = EEGNet(nb_classes=2, Chans=chans, Samples=samples,
    #                 dropoutRate=0.5, kernLength=kernLength, F1=F1, D=D, F2=F2,
    #                 dropoutType='Dropout')
    # model1 = ShallowConvNet(nb_classes=2, Chans=chans, Samples=samples, dropoutRate=0.5)
    model1 = cnn_best(nb_classes=2, Chans=X_train.shape[1], Samples=X_train.shape[2],
                      dropoutRate=0.5, kernLength=kernLength, C1=C1, C2=32, D1=D1)

    model1.load_weights(model_path + 'model.h5')
    probs = model.predict(X_test)
    y_pred = probs.argmax(axis=-1)
    acc = np.mean(y_pred == Y_test.argmax(axis=-1))
    print("Classification accuracy_lastmodel: %f " % (acc))

    # 画混淆矩阵
    print('y_te: ', y_te)
    print('y_pred: ', y_pred)
    print('accuracy of the test set:', acc)
    CM = confusion_matrix(y_te, y_pred, labels=[0, 1])
    print('混淆矩阵:\n', CM)  # 打印出来看看
    print('左手MI:', CM[0, 0] / np.sum(CM[0]), ' 右手MI:', CM[1, 1] / np.sum(CM[1]))

    return acc, CM


if __name__ == '__main__':
    # 读取文件
    # Hezhiren_2MI_20230725_origin  Wuhao_2MI_20230726_origin
    # zhiren_2MI_20230718T160630  adis_2MI_20230721T181907  junhan_blink_clench_20230721T173317
    file_name_ = './data/Wuhao_2MI_20230726_origin'
    X, y = read_data_edf(file_name_+'.edf')
    # X, y = read_data_txt(file_name_+'.txt')

    # np.save('data.npy', X)
    # np.save('labels.npy', y)
    #
    # X = np.load('data.npy')
    # y = np.load('labels.npy')

    X *= 1e6

    # 随机k_ran次
    k_ran = 50
    acc_ran = np.zeros(k_ran)  # 不同随机数的相应结果
    CM_ran = np.zeros((2, 2))  # confusion matrix with different fold
    for rann in range(k_ran):
        # train-test separation
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=rann, stratify=y)
        # FBCSP得到CSP投影信号
        X_train, X_test = FBCSP(X_train, X_test, Y_train)
        # 得到CSP方差特征再用SVM分类识别
        acc, CM = SVM_classify(X_train, X_test, Y_train, Y_test)
        # 通过CNN分类识别
        # acc, CM = CNN_classify(X_train, X_test, Y_train, Y_test)
        acc_ran[rann] = acc
        CM_ran += CM.copy()

    # 将各个ran准确率相加，输出平均准确率
    acc_ran_avg = np.average(acc_ran)
    print('acc across rans:')
    print(acc_ran)
    print('average accuracy across rans:', acc_ran_avg)

    print('confusion matrix number-wise:')
    print(CM_ran)
    print('confusion matrix percent-wise:')
    print(CM_ran / np.sum(CM_ran[0]))
    print('accuracy:')
    for ii in range(2):
        print(CM_ran[ii, ii] / np.sum(CM_ran[ii]))
    CM_ran_acc = np.zeros((2, 2))
    for cla in range(2):
        CM_ran_acc[cla] = CM_ran[cla] / np.sum(CM_ran[cla])
    sns.set()
    f, ax = plt.subplots()
    sns.heatmap(CM_ran_acc, annot=True, ax=ax)  # 画热力图

    ax.set_title('confusion matrix, acc=' + str(acc_ran_avg))  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.show()