import numpy as np
import mne
from mne.decoding import CSP
import pywt
import joblib
from sklearn import svm
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal


def cspp(x_train, x_test, y_train):  # x_test,
    # 初始化
    # x_train：trials*channels*sampling_points
    # x_train *= 1e6
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
    csp_train_feature = np.zeros((trials_train, 2*filter_pairs))
    csp_test_feature = np.zeros((trials_test, 2*filter_pairs))
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

    for t in range(trials_test):
        # projecting the data onto the CSP filters
        projected_trial_test = np.dot(Filter, x_test[t, :, :])
        # generating the features as the log variance of the projected signals
        variances_test = np.var(projected_trial_test, axis=1)
        for f in range(variances_test.shape[0]):
            csp_test_feature[t, f] = np.log(variances_test[f] / variances_test.sum())

    return csp_train_feature, csp_test_feature


def frequency_filtering(signal_x, low_freq, high_freq, fs):
    # new_filtering = np.zeros(signal_x.shape)
    nyq = fs * 0.5  # 奈奎斯特频率
    # b, a = signal.cheby2(10, 40, [low_freq / nyq, high_freq / nyq], 'bandpass', analog=False)  # 10阶切比雪夫和FIR结果近似（8-30Hz）
    b, a = signal.butter(3, [low_freq / nyq, high_freq / nyq], 'bandpass', analog=False)  # 3阶巴特沃斯和FIR结果近似（8-30Hz）
    new_filtering = signal.filtfilt(b, a, signal_x)
    # for i in range(signal_x.shape[0]):
    #     # Wn=[low_freq/nyq, high_freq/nyq]截止频率/信号频率(奈奎斯特采样频率的一半)
    #     new_filtering[i] = signal.filtfilt(b, a, signal_x[i])
    return new_filtering


def bandpass_design(start_freq,end_freq,samp_freq,order):
    nqs_freq = samp_freq * 0.5
    low = start_freq / nqs_freq
    high = end_freq / nqs_freq
    sos = butter(order, [low,high], analog = False, btype = 'band',output = 'sos')
    return sos


def bandpass_filter(signal):
    sos = bandpass_design(1,40,200,order = 4)
    output = sosfilt(sos,signal)
    return output


if __name__ == '__main__':
    time_start = time.time()
    # Process data
    avg_sub_40_14_80trial_120length = []
    std_sub_40_14_80trial_120length = []
    for sub in ['A08T', 'A02T', 'A03T', 'A04T', 'A05T', 'A06T', 'A07T', 'A08T', 'A09T']:
        # 'A01T', 'A02T', 'A03T', 'A04T', 'A05T', 'A06T', 'A07T', 'A08T', 'A09T'
        # 'A01E', 'A02E', 'A03E', 'A04E', 'A05E', 'A06E', 'A07E', 'A08E', 'A09E'
        raw = mne.io.read_raw_gdf('./BCI Competition IV/' + sub + '.gdf', preload=True)
        # raw.plot()
        raw.set_channel_types({'EOG-left': 'eog', 'EOG-central': 'eog', 'EOG-right': 'eog'})

        raw.annotations.description[np.where(raw.annotations.description == '1023')] = 'bad'

        # Clean channel names to be able to use a standard 1005 montage
        new_names = {'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2', 'EEG-4': 'FC4',
                     'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4',
                     'EEG-8': 'C6', 'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4',
                     'EEG-14': 'P1', 'EEG-Pz': 'Pz', 'EEG-15': 'P2', 'EEG-16': 'POz'}
        raw.rename_channels(new_names)
        # raw.plot(n_channels=25, duration=10, start=1253)

        # use the average of all channels as reference
        raw.set_eeg_reference(ref_channels='average', projection=True)  # needed for inverse modeling
        # raw.plot(n_channels=5, duration=10, start=1253)  # , duration=3, color=dict(eeg='k')

        # load standard 10-20 montage
        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')

        # Read and set the EEG electrode locations
        raw.set_montage(ten_twenty_montage)

        # # raw filtering
        # freq_win = (8, 30)
        # raw.filter(freq_win[0], freq_win[1])

        # # 去除伪迹（独立成分分析ICA）
        # set up and fit the ICA
        ica = mne.preprocessing.ICA(n_components=15, random_state=0)  # n_components=20, max_iter=1000, random_state=97
        # 'random_state': a random seed that we get identical results each time when ICA is built by our servers
        ica.fit(raw)
        # ica.plot_sources(raw, start=1253, stop=1263)
        # ica.plot_components()
        # ica.plot_overlay(raw, exclude=[1, 4], picks='eeg')  # need to determine every time
        ica.exclude = [1, 4]  # need to determine every time
        ica.apply(raw)

        # raw.plot(n_channels=5, duration=10, start=1253)
        raw.plot_psd(fmax=40, picks='eeg', proj=True, dB=True, estimate='amplitude')
        # raw filtering
        freq_win = (8, 30)
        # filter_params = mne.filter.create_filter(raw.get_data(picks='eeg'), raw.info['sfreq'],
        #                                          l_freq=freq_win[0], h_freq=freq_win[1])
        # mne.viz.plot_filter(filter_params, raw.info['sfreq'], flim=(0, 100), fscale='linear', alim=(-300, 10))
        raw.filter(freq_win[0], freq_win[1])

        # raw.plot(n_channels=22, duration=10, start=1253)
        raw.plot_psd(fmax=40, picks='eeg', proj=True, dB=True, estimate='amplitude')
        events_from_annot, event_dict = mne.events_from_annotations(raw)
        print(event_dict)

        # # !!!events for training data!!!
        event_dicts = {'769': 6, '770': 7}  # , '771': 8, '772': 9
        if sub == 'A04T':
            event_dicts = {'769': 4, '770': 5}  # , '771': 6, '772': 7

        # # !!!events for testing data!!!
        # event_dicts = {'783': 6}

        task = {'769': 'Left_Hand', '770': 'Right_Hand'}  # , '771': 'Foot', '772': 'Tongue'

        t_min = -1
        t_max = 2

        epochs = mne.Epochs(raw, events_from_annot, event_dicts, picks='eeg', reject_by_annotation=True,
                            tmin=t_min, tmax=t_max, preload=True)  # baseline=(t_min, 0) baseline=(None, None)
        events_from_annot[:, 2] = events_from_annot[:, 2] - 5
        epochs.plot(events=events_from_annot, n_epochs=10, n_channels=22, event_id={'left': 1, 'right': 2})
        a=1

        # # 第二天数据A0xE重造epoch
        # index = np.concatenate((np.where(np.load('./model_2008/' + sub[:3] + '_label_te.npy') == 0),
        #                         np.where(np.load('./model_2008/' + sub[:3] + '_label_te.npy') == 1)),
        #                        axis=1)[0]
        # index = sorted(index)
        # events_data = np.load('./model_2008/' + sub[:3] + '_label_te.npy')[index] + 1
        # epochs_data = epochs.get_data().copy()[index]
        #
        # sfreq = 250  # 采样率
        # trials = epochs_data.shape[0]  # 数据包含的试次数量
        # channels = epochs_data.shape[1]  # 脑电数据的通道数
        # points = epochs_data.shape[2]  # 每个trial的点数
        # events = np.zeros((trials, 3))  # 初始化‘事件’，为了后续将数据转为epoch
        # for i in range(trials):  # 以trial为单位进行操作
        #     events[i] = [int(points * i), 0, int(events_data[i])]  # 默认的events创建格式（具体见MNE官网）
        # events = events.astype(int)  # 转换数据类型为int
        # ch_types = 'eeg'  # 设置通道类型
        # ch_names = epochs.ch_names
        #
        # info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)  # 创建相关信息（默认操作，见官网）
        # # 创建epoch
        # event_dicts = {'left': 1, 'right': 2}
        # epochs = mne.EpochsArray(epochs_data.copy(), info=info, events=events, tmin=t_min, #baseline=(t_min, 0),
        #                          event_id=event_dicts)  # , 'foot': 3, 'tongue': 4
        # # epochs.set_eeg_reference(ref_channels='average', projection=True)
        # # epochs.apply_proj()
        # ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        # epochs.set_montage(ten_twenty_montage)


        # # butterworth filtering
        # X_train = frequency_filtering(epochs.get_data()[:,:,:-1], 4, 40, fs=250)

        # # 将训练数据保存为npy格式
        # np.save('./bci_competition_IV_2a_nofilter_4s/' + sub + '_tr_data.npy', epochs.get_data()[:,:,:-1])
        # if sub == 'A04T':
        #     np.save('./bci_competition_IV_2a_nofilter_4s/' + sub + '_tr_labels.npy', epochs.events[:, 2]-3)
        # else:
        #     np.save('./bci_competition_IV_2a_nofilter_4s/' + sub + '_tr_labels.npy', epochs.events[:, 2]-5)

        # # 将测试数据保存为npy格式
        # np.save('./bci_competition_IV_2a_nofilter_4s/' + sub + '_te_data.npy', epochs.get_data()[:, :, :-1])
        # np.save('./bci_competition_IV_2a_nofilter_4s/' + sub + '_te_labels.npy', np.load('./model_2008/' + sub[:3] + '_label_te.npy') + 1)

        # # 将测试数据保存为npy格式 for graduation crossfold!!!
        # np.save('./graduation_design_results/5foldcrossval_dataset_bci_competition_IV2a_830filter_base1sMI2s/'
        #         + sub + '_test_data.npy', epochs.get_data()[:, :, :-1])
        # np.save('./graduation_design_results/5foldcrossval_dataset_bci_competition_IV2a_830filter_base1sMI2s/'
        #         + sub + '_test_labels.npy', np.load('./model_2008/' + sub[:3] + '_label_te.npy') + 1)

        # epochs.plot()
        # mne.concatenate_epochs()  # 多被试合并的操作

        # 自己的CSP代码
        # w_csp = CSP(n_components=4, reg='empirical')
        # # 得到空间滤波器w_csp以及每个trial/epoch的特征
        # epo = epochs['769', '770']
        # epo.filter(8, 30)
        # label = epo.events[:, 2].copy()
        # epos = epo.get_data().copy()
        # a = cspp(epos, label)
        # b = w_csp.fit_transform(epos, label)

        for tr_num in [26]:  # [26] [40]
            csp_svm_tband = []
            for interval in [14]:
                # OVR-FBCSP
                sfreq = 250  # 降采样率
                classes = list(event_dicts)
                n_classes = len(classes)  # 分类类别
                n_components = 4  # CSP特征选取个数
                t_band = 1  # 针对不同的时间段提取特征，0.5-1.5, 1-2, 1.5-2.5, 2-3, 2.5-3.5, 3-4
                filt = 1  # 滤波器组的个数10:（8Hz-30Hz，间隔4Hz，重叠2Hz）; 16:（4Hz-38Hz，间隔4Hz，重叠2Hz）
                if n_classes == 2:
                    n_classes_ = 1
                else:
                    n_classes_ = n_classes
                k_num = int(filt * n_classes_ * n_components * 1)  # 0.5 1 # 特征选择的个数 * t_band
                trials = epochs._data.shape[0]  # 数据包含的试次数量
                channels = epochs._data.shape[1]  # 脑电数据的通道数
                points = epochs._data.shape[2]  # 每个trial的点数
                k_fold = 1  # k折交叉验证（这里的k_fold与我们数据大小有关，一定要能够被类试次整除）
                num_fold = len(epochs[classes[0]])
                for ii in range(n_classes - 1):
                    num_fold = min(num_fold, len(epochs[classes[ii + 1]]))
                num_fold = int(num_fold / k_fold)  # int 取整数（也就是往下取整）
                k_ran = 10  # 设置k_ran个不同的随机数，进行k_ran次k_fold折(k_ran×k_fold)交叉验证
                acc_tb_ran = []
                CM_fold_ran = np.zeros((n_classes, n_classes))  # confusion matrix with different fold and different ran

                for rann in range(k_ran):
                    np.random.seed(rann * 10 + 5)  # 设置随机数（用于将数据和标签（事件）同时打乱）
                    epochs_mov = []
                    for n_cla in range(n_classes):
                        epochs_mov.append(np.zeros((filt, len(epochs[classes[n_cla]]), channels, points)))
                    for freq in range(filt):
                        # # one filter
                        epochs_filt = epochs.copy()
                        # # multi filters
                        # freq_win = (8 + freq * 2, 12 + freq * 2)
                        # epochs_filt = epochs.copy().filter(freq_win[0], freq_win[1], verbose=False)
                        for n_cla in range(n_classes):
                            # 将滤完波后的epochs_filt的每一类拿出来，赋予epochs_mov
                            epochs_mov[n_cla][freq] = epochs_filt[classes[n_cla]].copy().get_data()

                    # 将epochs_mov打乱
                    for n_cla in range(n_classes):
                        ran = np.random.permutation(epochs_mov[n_cla].shape[1])  # 得到随机数列
                        epochs_mov[n_cla] = epochs_mov[n_cla][:, ran].copy()  # 打乱
                    acc_tb = []
                    for t_b in range(t_band):
                        t_start = 0.5
                        # t_end = 0.6
                        # t_start = 3 + 0.1 * t_b
                        t_end = t_start + 0.1 * (interval+1)
                        acc_fold = np.zeros(k_fold)  # 不同随机数的相应结果
                        CM_fold = np.zeros((n_classes, n_classes))  # confusion matrix with different fold
                        for fold in range(k_fold):
                            te = []
                            tr = []
                            for n_cla in range(n_classes):
                                # 得到当前fold的训练和测试数据集的下标（最后k_fold * num_fold:-1作为测试集，前面的数据五等分，保证不同类训练数据一样多）
                                tr.append(np.arange(epochs_mov[n_cla].shape[1])[:tr_num])
                                te.append(np.arange(epochs_mov[n_cla].shape[1])[tr_num:])

                                if n_cla == 0:
                                    label_tr, label_te = np.ones(len(tr[n_cla])).astype(int), np.ones(len(te[n_cla])).astype(int)
                                    num_tr, num_te = len(tr[n_cla]), len(te[n_cla])
                                else:
                                    label_tr = np.concatenate((label_tr, np.ones(len(tr[n_cla])) + n_cla)).astype(int)
                                    label_te = np.concatenate((label_te, np.ones(len(te[n_cla])) + n_cla)).astype(int)
                                    num_tr += len(tr[n_cla])
                                    num_te += len(te[n_cla])
                            if n_classes == 2:
                                n_classes_ = 1
                            else:
                                n_classes_ = n_classes
                            csp_TR = np.zeros((num_tr, filt, n_components * n_classes_))  # 最后得到的csp特征，试次*频段*特征数目
                            csp_TE = np.zeros((num_te, filt, n_components * n_classes_))  # 最后得到的csp特征，试次*频段*特征数目
                            for fil in range(filt):
                                for n_cla in range(n_classes):
                                    if n_cla == 0:
                                        epochs_tr = epochs_mov[n_cla][fil, tr[n_cla], :,
                                                    int(sfreq * (t_start-t_min)):int(sfreq * (t_end-t_min))].copy()
                                        epochs_te = epochs_mov[n_cla][fil, te[n_cla], :,
                                                    int(sfreq * (t_start-t_min)):int(sfreq * (t_end-t_min))].copy()
                                    else:
                                        epochs_tr = np.concatenate((epochs_tr,
                                                                    epochs_mov[n_cla][fil, tr[n_cla], :,
                                                                    int(sfreq * (t_start-t_min)):int(sfreq * (t_end-t_min))]))
                                        epochs_te = np.concatenate((epochs_te,
                                                                    epochs_mov[n_cla][fil, te[n_cla], :,
                                                                    int(sfreq * (t_start-t_min)):int(sfreq * (t_end-t_min))]))

                                # # # 保存交叉验证数据
                                # np.save('./graduation_design_results/5foldcrossval_dataset_bci_competition_IV2a_830filter_base1sMI2s/'
                                #         + sub + '_fold_' + str(fold) + '_tr_data.npy', epochs_tr)
                                # np.save('./graduation_design_results/5foldcrossval_dataset_bci_competition_IV2a_830filter_base1sMI2s/'
                                #         + sub + '_fold_' + str(fold) + '_te_data.npy', epochs_te)
                                # np.save('./graduation_design_results/5foldcrossval_dataset_bci_competition_IV2a_830filter_base1sMI2s/'
                                #         + sub + '_fold_' + str(fold) + '_tr_labels.npy', label_tr)
                                # np.save('./graduation_design_results/5foldcrossval_dataset_bci_competition_IV2a_830filter_base1sMI2s/'
                                #         + sub + '_fold_' + str(fold) + '_te_labels.npy', label_te)

                                for ovr in range(n_classes):
                                    label_tr_ovr = label_tr.copy()
                                    label_tr_ovr[np.where(label_tr_ovr == ovr + 1)] = 0  # 把一类看作一类
                                    label_tr_ovr[np.where(label_tr_ovr != 0)] = 1  # 把另外五类看作另一类

                                    # # 库函数
                                    w_csp = CSP(n_components=n_components, reg='empirical')
                                    # 得到空间滤波器w_csp以及每个trial/epoch的特征
                                    csp_TR[:, fil, ovr * n_components: (ovr + 1) * n_components] = w_csp.fit_transform(epochs_tr, label_tr_ovr).copy()
                                    # w_csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
                                    # clf.append(w_csp)  # 将空间滤波器进行存储，以便后续测试集使用
                                    # 将空间滤波器应用在测试集数据上，得到测试特征
                                    csp_TE[:, fil, ovr * n_components: (ovr + 1) * n_components] = w_csp.transform(epochs_te).copy()

                                    # # self-CSP
                                    # a, b = cspp(epochs_tr, epochs_te, label_tr_ovr)
                                    # csp_TR[:, fil, ovr * n_components: (ovr + 1) * n_components] = a
                                    # csp_TE[:, fil, ovr * n_components: (ovr + 1) * n_components] = b

                                    if n_classes == 2:
                                        break

                            csp_tr = np.reshape(csp_TR, (csp_TR.shape[0], csp_TR.shape[1] * csp_TR.shape[2]))
                            csp_te = np.reshape(csp_TE, (csp_TE.shape[0], csp_TR.shape[1] * csp_TE.shape[2]))

                            # # 初始化selector（多种score_func可选，此处列举两种）
                            selector = SelectKBest(score_func=mutual_info_classif, k=k_num)
                            # selector = SelectKBest(score_func=f_classif, k=k+1)
                            # 训练selector
                            selector.fit(csp_tr, label_tr)
                            # 得到训练集和测试集的最优特征
                            TransX_tr = selector.transform(csp_tr)
                            TransX_te = selector.transform(csp_te)

                            # 初始化SVM分类器
                            svm = SVC(decision_function_shape='ovr')
                            # 分类器训练
                            svm.fit(TransX_tr, label_tr)  # csp_tr TransX_tr
                            # 得到预测值
                            y_pred = svm.predict(TransX_te)  # csp_te TransX_te
                            print(list(y_pred))

                            # # SVM可视化
                            # h = 0.002
                            # x_min, x_max = TransX_tr[:, 0].min() - 0.1, TransX_tr[:, 0].max() + 0.1
                            # y_min, y_max = TransX_tr[:, 1].min() - 0.1, TransX_tr[:, 1].max() + 0.1
                            # # x_min, x_max = -0.1, 0.6
                            # # y_min, y_max = -0.1, 0.6
                            # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            #                      np.arange(y_min, y_max, h))
                            # Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
                            # Z = Z.reshape(xx.shape)
                            # # Plot the training points
                            # # plt.subplot(1, 2, 1)
                            # plt.scatter(f_rms_tr[:, 0], f_rms_tr[:, 1], c=label_tr)
                            # plt.contourf(xx, yy, Z, cmap=plt.cm.ocean, alpha=0.2)
                            # y_pred = svc.predict(f_rms_tr)
                            # # plt.subplot(1, 2, 2)
                            # # plt.scatter(f_rms_tr[:, 0], f_rms_tr[:, 1], c=y_pred)
                            # # plt.contourf(xx, yy, Z, cmap=plt.cm.ocean, alpha=0.2)
                            # plt.title('acc=' + str(accuracy_score(labels, y_pred)) + '  wrong nums=' + str(
                            #     int(len(labels) - accuracy_score(labels, y_pred) * len(labels))))
                            #
                            # # plt.ion()
                            # plt.show()
                            # # plt.pause(2)
                            # # plt.clf()  # 清除图像

                            # 将预测值与标签进行对比计算出分数
                            acc_fold[fold] = accuracy_score(label_te, y_pred)
                            # 画混淆矩阵
                            print('accuracy of the test set:', acc_fold[fold])
                            CM = confusion_matrix(label_te, y_pred, labels=list(np.arange(n_classes) + 1))  # 1, 2, 3, 4, 5, 6
                            print(CM)
                            for cm in range(n_classes):
                                print(classes[cm] + ':', CM[cm, cm] / np.sum(CM[cm]))
                            CM_fold += confusion_matrix(label_te, y_pred,
                                                        labels=list(np.arange(n_classes) + 1))  # 1, 2, 3, 4, 5, 6

                        # 将同一最优特征的各个fold准确率相加，输出最佳的特征选择数目k和在该数目下的平均准确率
                        acc_fold_avg = np.average(acc_fold)
                        print('acc across folds:')
                        print(acc_fold)
                        print('average accuracy across folds:', acc_fold_avg)

                        # print('confusion matrix number-wise:')
                        # print(CM_fold)
                        # print('confusion matrix percent-wise:')
                        # print(CM_fold / np.sum(CM_fold[0]))
                        # print('accuracy:')
                        # for ii in range(n_classes):
                        #     print(CM_fold[ii, ii] / np.sum(CM_fold[ii]))
                        # CM_fold_acc = np.zeros((n_classes, n_classes))
                        # for cla in range(n_classes):
                        #     CM_fold_acc[cla] = CM_fold[cla] / np.sum(CM_fold[cla])
                        # sns.set()
                        # f, ax = plt.subplots()
                        # sns.heatmap(CM_fold_acc, annot=True, ax=ax)  # 画热力图
                        #
                        # ax.set_title('confusion matrix, acc=' + str(acc_fold_avg))  # 标题
                        # ax.set_xlabel('predict')  # x轴
                        # ax.set_ylabel('true')  # y轴
                        # plt.show()

                        acc_tb.append(acc_fold_avg)

                    # plt.plot(acc_tb, linewidth=0.5)
                    acc_tb_ran.append(acc_tb)
                    CM_fold_ran += CM_fold

                acc_tb_ran_avg = np.average(acc_tb_ran)
                # plt.plot(acc_tb_ran_avg, linewidth=3, color='black')

                # # 计算std并保存
                std = 0
                for std_n in range(k_ran):
                    std += (acc_tb_ran[std_n][0] - acc_tb_ran_avg) ** 2
                std = np.sqrt(std / k_ran)
                avg_sub_40_14_80trial_120length.append(acc_tb_ran_avg)
                std_sub_40_14_80trial_120length.append(std)

                print('confusion matrix number-wise:')
                print(CM_fold_ran)
                print('confusion matrix percent-wise:')
                print(CM_fold_ran / np.sum(CM_fold_ran[0]))
                print('accuracy:')
                for ii in range(n_classes):
                    print(CM_fold_ran[ii, ii] / np.sum(CM_fold_ran[ii]))
                CM_fold_ran_acc = np.zeros((n_classes, n_classes))
                for cla in range(n_classes):
                    CM_fold_ran_acc[cla] = CM_fold_ran[cla] / np.sum(CM_fold_ran[cla])

                # sns.set()
                # f, ax = plt.subplots()
                # sns.heatmap(CM_fold_ran_acc, annot=True, ax=ax)  # 画热力图
                #
                # ax.set_title('confusion matrix, acc=' + str(acc_tb_ran_avg))  # 标题
                # ax.set_xlabel('predict')  # x轴
                # ax.set_ylabel('true')  # y轴
                # plt.show()
                csp_svm_tband.append([acc_tb_ran_avg, CM_fold_ran[0, 0] / np.sum(CM_fold_ran[0]),
                                      CM_fold_ran[1, 1] / np.sum(CM_fold_ran[1])])
            # joblib.dump(csp_svm_tband, './graduation_design_results/bci_competition_IV_2a_fast_recog/csp_svm/' + sub +
            #             '_tr_num_' + str(tr_num*2) + '_csp_svm_0_5_interval0_1')
    a = 1
