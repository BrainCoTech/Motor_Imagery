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


def csp_test(Filter, x_test):
    # 初始化
    # x_test：trials*channels*sampling_points
    trials_test = x_test.shape[0]
    # 计算特征矩阵
    filter_pairs = 2  # CSP特征选择参数m,CSP特征为2*m个
    csp_test_feature = np.zeros((trials_test, 2*filter_pairs))

    for t in range(trials_test):
        # projecting the data onto the CSP filters
        projected_trial_test = np.dot(Filter, x_test[t, :, :])
        # generating the features as the log variance of the projected signals
        variances_test = np.var(projected_trial_test, axis=1)
        for f in range(variances_test.shape[0]):
            csp_test_feature[t, f] = np.log(variances_test[f] / variances_test.sum())

    return csp_test_feature


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


def testing(file_name):
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
            if np.any(abs(data[:, index:index + epoch_length]) > 50):
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
    epoch_data = np.array(epoch_data)
    labels = np.array(labels).astype(int)

    # ************************************************ 画粗略分段波形图 ************************************************
    # 单图
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

    clf = joblib.load('model.m')

    tr_trials = int(trials * 0.75)  # trials
    features = 4  # 两类数据，每类数据2个特征，所以一个频带4个特征
    csp_TE = np.zeros((trials-tr_trials, filt, features))  # 最后得到的csp特征，试次*频段*特征数目

    epochs_filt = np.zeros((filt, trials-tr_trials, channels, points))

    # 用滤波器组对epoch数据进行滤波，得到epochs_filter
    for freq in range(filt):
        f_low = 8 + freq * 2  # low frequency
        f_high = 12 + freq * 2  # high frequency
        epochs_filt[freq, :] = butter_bandpass_filter(epoch_data[tr_trials:].copy(), f_low, f_high, fs)

    for fil in range(filt):
        epochs_fil_te = epochs_filt[fil, :, :, int(0.5*fs):int(3.5*fs)].copy()
        w_csp = clf[fil].copy()
        # 通过组合四个二分类滤波器得到四分类滤波器
        for ovr in range(1):
            # 得到空间滤波器w_csp以及每个trial/epoch的特征
            csp_TE[:, fil, ovr * 4:(ovr + 1) * 4] = csp_test(w_csp, epochs_fil_te)

    csp_te = np.reshape(csp_TE, (csp_TE.shape[0], filt * csp_TE.shape[2]))

    # # 得到训练集和测试集的最优特征
    selector = clf[-2]
    csp_te = selector.transform(csp_te)

    # 通过训练好的SVM得到预测值
    svm = clf[-1]
    y_pred = svm.predict(csp_te)

    # 将预测值与标签进行对比计算出分数
    label_test = labels[tr_trials:].copy()
    sco = accuracy_score(label_test, y_pred)
    print(sco)

    # 画混淆矩阵
    C2 = confusion_matrix(label_test, y_pred, labels=[1, 2])
    print(C2)  # 打印出来看看
    print(C2[0,0] / np.sum(C2[0]), C2[1,1] / np.sum(C2[1]))
    C2_acc = np.zeros((2, 2))
    for cla in range(2):
        C2_acc[cla] = C2[cla] / np.sum(C2[cla])
    sns.set()
    f, ax = plt.subplots()
    sns.heatmap(C2_acc, annot=True, ax=ax)  # 画热力图

    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.show()


def online_test(raw_data, state, move_data):
    move_data = np.array(move_data)
    y_pred = 0  # 预测值默认是0
    raw_data = np.array(raw_data)
    data = raw_data[:-1].copy()
    label = int(raw_data[-1].copy()[0])

    if state == 1 and move_data.shape[1] >= 3500:
        move_data = np.concatenate((move_data, data.copy()), axis=1)
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
        # emg_plot(data)
        # plt.show()

        # FBCSP
        filt = 10  # 滤波器组的个数（8Hz-30Hz，间隔4Hz，重叠2Hz）
        channels = move_data.shape[0]  # 脑电数据的通道数
        points = move_data.shape[1]  # 每个trial的点数

        clf = joblib.load('model.m')

        features = 4  # 两类数据，每类数据2个特征，所以一个频带4个特征
        csp_TE = np.zeros((1, filt, features))  # 最后得到的csp特征，试次*频段*特征数目

        epochs_filt = np.zeros((filt, 1, channels, points))

        # 用滤波器组对epoch数据进行滤波，得到epochs_filter
        for freq in range(filt):
            f_low = 8 + freq * 2  # low frequency
            f_high = 12 + freq * 2  # high frequency
            epochs_filt[freq, :] = butter_bandpass_filter(move_data.copy(), f_low, f_high, fs)

        for fil in range(filt):
            epochs_fil_te = epochs_filt[fil, :, :, int(0.5 * fs):int(3.5 * fs)].copy()
            w_csp = clf[fil].copy()
            # 通过组合四个二分类滤波器得到四分类滤波器
            for ovr in range(1):
                # 得到空间滤波器w_csp以及每个trial/epoch的特征
                csp_TE[:, fil, ovr * 4:(ovr + 1) * 4] = csp_test(w_csp, epochs_fil_te)

        csp_te = np.reshape(csp_TE, (csp_TE.shape[0], filt * csp_TE.shape[2]))

        # # 得到训练集和测试集的最优特征
        selector = clf[-2]
        csp_te = selector.transform(csp_te)

        # 通过训练好的SVM得到预测值
        svm = clf[-1]
        y_pred = svm.predict(csp_te)

        print('y_pred: ', y_pred)
        print('label: ', label)

        # 将预测值与标签进行对比
        if y_pred[0] == label:
            print('correct!')
        else:
            print('wrong~')
        move_data = move_data[:, -1:].copy()
        state = 0

    elif state == 1 and move_data.shape[1] < 3500:
        move_data = np.concatenate((move_data, data.copy()), axis=1)

    else:
        move_data = data.copy()
        state = 1

    return [state, move_data, y_pred]


if __name__ == '__main__':
    # 读取文件
    # junhan_2MI_20230717T115414 adis_2MI_20230721T181907
    file_name_ = './data/adis_2MI_20230721T181907.txt'
    testing(file_name_)