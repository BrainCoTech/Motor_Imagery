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
import mne


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
    raw = mne.io.read_raw_edf(file_name, preload=True)
    # Clean channel names to be able to use a standard 1005 montage
    new_names = {'EEG Fp1-REF': 'FC5', 'EEG Fp2-REF': 'FC3', 'EEG F7-REF': 'FC1', 'EEG F3-REF': 'FCz',
                 'EEG Fz-REF': 'FC2', 'EEG F4-REF': 'FC4', 'EEG F8-REF': 'FC6',
                 'EEG A1-REF': 'C5', 'EEG T3-REF': 'C3', 'EEG C3-REF': 'C1', 'EEG Cz-REF': 'Cz',
                 'EEG C4-REF': 'C2', 'EEG T4-REF': 'C4', 'EEG A2-REF': 'C6',
                 'EEG T5-REF': 'CP5', 'EEG P3-REF': 'CP3', 'EEG Pz-REF': 'CP1', 'EEG P4-REF': 'CPz',
                 'EEG T6-REF': 'CP2', 'EEG O1-REF': 'CP4', 'EEG O2-REF': 'CP6'}
    raw.rename_channels(new_names)
    # raw.plot(block=True, scalings=1e-4)

    # raw.set_eeg_reference(ref_channels=['EEG ROC-REF', 'EEG LOC-REF'])
    # raw.apply_proj()
    # raw.plot(block=True, scalings=1e-4)

    raw.info['bads'] = ['EEG ROC-REF', 'EEG LOC-REF', 'ECG EKG-REF', 'Photic-REF', 'IBI', 'Bursts', 'Suppr']

    # raw.info['bads'] = ['EEG ROC-REF', 'EEG LOC-REF', 'ECG EKG-REF', 'Photic-REF', 'IBI', 'Bursts', 'Suppr',
    #                     'FC5', 'FC1', 'FCz', 'FC2', 'FC6', 'Cz', 'CPz', 'CP6', 'CP5', 'CP2', 'CP1']

    ch_names = list(set(raw.ch_names) - set(raw.info['bads']))  # 得到需要的channels(set会重排顺序)
    ch_names.sort(key=raw.ch_names.index)  # 排序为raw.ch_names的导联顺序

    # ref
    # new_names = {'EEG Fp1-LOCROC': 'FC5', 'EEG Fp2-LOCROC': 'FC3', 'EEG F7-LOCROC': 'FC1', 'EEG F3-LOCROC': 'FCz',
    #              'EEG Fz-LOCROC': 'FC2', 'EEG F4-LOCROC': 'FC4', 'EEG F8-LOCROC': 'FC6',
    #              'EEG A1-LOCROC': 'C5', 'EEG T3-LOCROC': 'C3', 'EEG C3-LOCROC': 'C1', 'EEG Cz-LOCROC': 'Cz',
    #              'EEG C4-LOCROC': 'C2', 'EEG T4-LOCROC': 'C4', 'EEG A2-LOCROC': 'C6',
    #              'EEG T5-LOCROC': 'CP5', 'EEG P3-LOCROC': 'CP3', 'EEG Pz-LOCROC': 'CP1', 'EEG P4-LOCROC': 'CPz',
    #              'EEG T6-LOCROC': 'CP2', 'EEG O1-LOCROC': 'CP4', 'EEG O2-LOCROC': 'CP6'}
    # raw.rename_channels(new_names)

    # # raw filtering
    freq_win = (4, 40)
    raw.filter(freq_win[0], freq_win[1])

    events_from_annot, event_dict = mne.events_from_annotations(raw)
    print(event_dict)
    event_dicts = {'LeftMI': 3, 'RightMI': 4}

    t_min = 0.5
    t_max = 3

    epochs = mne.Epochs(raw, events_from_annot, event_dicts, picks=ch_names, reject_by_annotation=True,
                        tmin=t_min, tmax=t_max, baseline=(None, None), preload=True)
    # epochs.plot()

    epoch_data = epochs.get_data() * 1e6
    labels = epochs.events[:, 2] - 3

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

    # FBCSP1
    fs = 500
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
        epochs_fil_tr = epochs_filt[fil, :, :, :].copy()
        # 通过组合四个二分类滤波器得到四分类滤波器
        for ovr in range(1):
            # 得到空间滤波器w_csp以及每个trial/epoch的特征
            [w_csp, a] = csp_train(epochs_fil_tr, labels[:tr_trials])
            csp_TR[:, fil, ovr * 4:(ovr + 1) * 4] = a.copy()
            clf.append(w_csp)  # 将空间滤波器进行存储，以便后续测试集使用

    csp_tr = np.reshape(csp_TR, (csp_TR.shape[0], filt * csp_TR.shape[2]))

    # # 初始化selector（多种score_func可选，此处列举两种）
    # k_best = features * filt  # features * filt
    # selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
    # # selector = SelectKBest(score_func=f_classif, k=k+1)
    # # 训练selector
    # selector.fit(csp_tr, labels.copy()[:tr_trials])
    # clf.append(selector)
    # # 得到训练集和测试集的最优特征
    # csp_tr = selector.transform(csp_tr)

    # 初始化SVM分类器
    svm = SVC(decision_function_shape='ovr')
    # 分类器训练
    svm.fit(csp_tr, labels.copy()[:tr_trials])

    clf.append(svm)

    # joblib.dump(clf, './model/fbcsp_0_64trials_8_30Hz_10filts_svm_0_5__3_5_' + str(k_best) + 'kbest_clf.m', True)
    joblib.dump(clf, 'model.m', True)


if __name__ == '__main__':
    # 读取文件
    # Hezhiren_2MI_20230725_origin Hezhiren_2MI_20230725_ref Wuhao_2MI_20230726_origin
    file_name_ = './data/Wuhao_2MI_20230726_origin.edf'
    training(file_name_)