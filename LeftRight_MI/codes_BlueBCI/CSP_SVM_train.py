import numpy as np
import joblib
from sklearn import svm
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal


def csp_train(x_train, y_train):
    # 初始化
    # x_train：trials*channels*sampling_points
    # x_train *= 1e6
    trials_train = x_train.shape[0]
    # trials_test = x_test.shape[0]
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
    # 矩阵白化
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
    # d0 = np.linalg.eig(transformed_cov0)[0].real
    # d00 = np.linalg.eig(transformed_cov0)[0]#.real
    # dt01 = np.diag(d0) + np.diag(d1)
    # 计算投影矩阵W
    csp_matrix = np.dot(u1.T, p)
    # 计算特征矩阵
    filter_pairs = 2  # CSP特征选择参数m,CSP特征为2*m个
    csp_train_feature = np.zeros((trials_train, 2*filter_pairs))
    # csp_test_feature = np.zeros((trials_test, 2*filter_pairs))
    Filter = np.vstack((csp_matrix[0:filter_pairs, :], csp_matrix[-filter_pairs:, :]))
    # extracting the CSP features from each trial
    for t in range(trials_train):
        # projecting the data onto the CSP filters
        # ptt = np.zeros((1, 2, x_train.shape[2]))
        projected_trial_train = np.dot(Filter, x_train[t, :, :])
        # # 频谱测试效果
        # ptt[0] = projected_trial_train[[0, -1], :].copy()
        #
        # sfreq1 = 250
        # ch_types1 = 'eeg'
        # ch_names1 = ['C3', 'C4']
        # info1 = mne.create_info(ch_names=ch_names1, sfreq=sfreq1, ch_types=ch_types1)
        # epochss = mne.EpochsArray(ptt, info=info1)
        #
        # epochs_train_data1[t].plot_psd(fmax=50)
        # epochss.plot_psd(fmax=50)

        # generating the features as the log variance of the projected signals
        variances_train = np.var(projected_trial_train, axis=1)
        for f in range(variances_train.shape[0]):
            csp_train_feature[t, f] = np.log(variances_train[f] / variances_train.sum())

    return Filter, csp_train_feature


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
    # # plot frequency response
    # w, h = signal.freqz(b, a, worN=2000)
    # plt.plot(w, abs(h), label="order = %d" % order)
    # # # 滤波器的二阶截面表示
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


def training(file_name):
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
    # fs = 500
    # data = data[:, ::int(1000 / fs)]
    # label_row = label_row[::int(1000 / fs)]

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
            if np.any(abs(data[:, index:index+epoch_length]) > 50):
                pass
            else:
                epoch_data.append(data[:, index:index+epoch_length])
                labels.append(label_row[index])
                # ******以下代码画图用******
                if flag == 0:
                    epoch_data_ = data[:, index:index+epoch_length].copy()
                    flag = 1
                else:
                    epoch_data_ = np.concatenate((epoch_data_, data[:, index:index+epoch_length].copy()), axis=1)
                # ******以上代码画图用******
            # index往后移epoch_length个点
            index += epoch_length
        index += 1
    epoch_data = np.array(epoch_data)
    labels = np.array(labels).astype(int)

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

    # FBCSP
    filt = 10  # 滤波器组的个数（8Hz-30Hz，间隔4Hz，重叠2Hz）
    trials = epoch_data.shape[0]  # 数据包含的试次数量
    channels = epoch_data.shape[1]  # 脑电数据的通道数
    points = epoch_data.shape[2]  # 每个trial的点数

    clf = []  # 保存一些参数及分类模型
    tr_trials = int(trials * 0.75)  # trials
    features = 4  # 两类数据，每类数据2个特征，所以一个频带4个特征
    csp_TR = np.zeros((tr_trials, filt, features))  # 最后得到的csp特征，试次*频段*特征数目

    epochs_filt = np.zeros((filt, tr_trials, channels, points))

    # 用滤波器组对epoch数据进行滤波，得到epochs_filter
    for freq in range(filt):
        f_low = 8 + freq * 2  # low frequency
        f_high = 12 + freq * 2  # high frequency
        epochs_filt[freq, :] = butter_bandpass_filter(epoch_data[:tr_trials].copy(), f_low, f_high, fs)

    for fil in range(filt):
        epochs_fil_tr = epochs_filt[fil, :, :, int(0.5*fs):int(3.5*fs)].copy()
        # 通过组合四个二分类滤波器得到四分类滤波器
        for ovr in range(1):
            # 得到空间滤波器w_csp以及每个trial/epoch的特征
            [w_csp, a] = csp_train(epochs_fil_tr, labels[:tr_trials])
            csp_TR[:, fil, ovr * 4:(ovr + 1) * 4] = a.copy()
            clf.append(w_csp)  # 将空间滤波器进行存储，以便后续测试集使用

    csp_tr = np.reshape(csp_TR, (csp_TR.shape[0], filt * csp_TR.shape[2]))

    # 初始化selector（多种score_func可选，此处列举两种）
    k_best = features * filt  # features * filt
    selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
    # selector = SelectKBest(score_func=f_classif, k=k+1)
    # 训练selector
    selector.fit(csp_tr, labels.copy()[:tr_trials])
    clf.append(selector)
    # 得到训练集和测试集的最优特征
    csp_tr = selector.transform(csp_tr)

    # 初始化SVM分类器
    svm = SVC(decision_function_shape='ovr')
    # 分类器训练
    svm.fit(csp_tr, labels.copy()[:tr_trials])

    clf.append(svm)

    # joblib.dump(clf, './model/fbcsp_0_64trials_8_30Hz_10filts_svm_0_5__3_5_' + str(k_best) + 'kbest_clf.m', True)
    joblib.dump(clf, 'model.m', True)


if __name__ == '__main__':
    # 读取文件
    # junhan_2MI_20230717T115414 adis_2MI_20230721T181907
    file_name_ = './data/adis_2MI_20230721T181907.txt'
    training(file_name_)