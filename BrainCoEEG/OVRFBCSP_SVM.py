import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

import os
import yaml

from MI_preprcessing import data_preprocess

current_directory = os.path.dirname(os.path.abspath(__file__))

with open(f'{current_directory}/settings.yaml', 'r') as f:
    settings_config = yaml.load(f, Loader=yaml.FullLoader)
fs = settings_config['fs']  # 采样率
n_classes = settings_config['n_classes']  # 类别个数
n_filters = settings_config['n_filters']  # 滤波器组的个数（8Hz-30Hz，间隔4Hz，重叠2Hz）
n_components = settings_config['n_components']  # CSP特征选取个数
k_fold = settings_config['k_fold']  # k折交叉验证（这里的k_fold与我们数据大小有关，一定要能够被类试次整除）
k_ran = settings_config['k_ran']  # 设置k_ran个不同的随机数，进行k_ran次k_fold折(k_ran×k_fold)交叉验证


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = fs * 0.5  # 奈奎斯特采样频率
    low, high = lowcut / nyq, highcut / nyq
    # # # 滤波器的二阶截面表示
    sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
    # filtering
    y = signal.sosfiltfilt(sos, data)

    return y


def get_csp_feature(csp_proj):
    # extracting the CSP features from each trial
    csp_feature = np.zeros((csp_proj.shape[0], csp_proj.shape[1]))
    for t, xx in enumerate(csp_proj):
        # generating the features as the log variance of the projected signals
        variances_x = np.var(xx, axis=1)
        for f in range(variances_x.shape[0]):
            csp_feature[t, f] = np.log(variances_x[f] / variances_x.sum())

    return csp_feature

 
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
    filter_pairs = int(n_components/2)  # CSP特征选择参数，CSP特征为n_components个
    Filter = np.vstack((csp_matrix[0:filter_pairs, :], csp_matrix[-filter_pairs:, :]))

    # get the CSP projected signals
    csp_projected_tr = np.zeros((trials_train, n_components, x_train.shape[2]))
    csp_projected_te = np.zeros((trials_test, n_components, x_test.shape[2]))
    x_train_test = [x_train, x_test]
    for flag, x in enumerate(x_train_test):
        for t, xx in enumerate(x):
            # projecting the data onto the CSP filters
            if flag == 0:
                csp_projected_tr[t] = np.dot(Filter, xx)
            else:
                csp_projected_te[t] = np.dot(Filter, xx)

    return csp_projected_tr, csp_projected_te


def FB_filter(X):
    trials = X.shape[0]  # 训练数据包含的试次数量
    channels = X.shape[1]  # 每个trial的通道数
    points = X.shape[2]  # 每个trial的点数

    epochs_filt = np.zeros((trials, n_filters, channels, points))

    # 用滤波器组对epoch数据进行滤波并得到CSP投影信号
    for filt in range(n_filters):
        f_low = 4 + filt * 2  # low frequency
        f_high = 8 + filt * 2  # high frequency
        epochs_filt[:, filt, :, :] = butter_bandpass_filter(X, f_low, f_high, fs)

    return epochs_filt


def OVRFBCSP(X_tr, X_te, Y_tr):
    tr_trials = X_tr.shape[0]  # 训练数据包含的试次数量
    te_trials = X_te.shape[0]  # 测试数据包含的试次数量
    points = X_tr.shape[2]  # 每个trial的点数

    X_filt_tr = FB_filter(X_tr)
    X_filt_te = FB_filter(X_te)

    csp_proj_tr = np.zeros((tr_trials, n_filters, n_classes*n_components, points))
    csp_proj_te = np.zeros((te_trials, n_filters, n_classes*n_components, points))

    for filt in range(n_filters):
        epochs_filt_tr = X_filt_tr[:, filt, :, :].copy()
        epochs_filt_te = X_filt_te[:, filt, :, :].copy()

        for ovr in range(n_classes):
            Y_tr_ovr = Y_tr.copy()
            Y_tr_ovr[np.where(Y_tr_ovr == ovr + 1)] = 0  # 把一类看作一类
            Y_tr_ovr[np.where(Y_tr_ovr != 0)] = 1  # 把其他类看作另一类

            # 得到CSP投影信号，csp_porj_tr：试次数目*10个滤波器*（n_classes*2个特征）*时间点数
            (csp_proj_tr[:, filt, ovr*n_components:(ovr+1)*n_components, :],
             csp_proj_te[:, filt, ovr*n_components:(ovr+1)*n_components, :]) =\
                csp_projected(epochs_filt_tr, epochs_filt_te, Y_tr_ovr)

    csp_proj_tr = np.reshape(csp_proj_tr, (csp_proj_tr.shape[0], csp_proj_tr.shape[1] * csp_proj_tr.shape[2], csp_proj_tr.shape[3]))
    csp_proj_te = np.reshape(csp_proj_te, (csp_proj_te.shape[0], csp_proj_te.shape[1] * csp_proj_te.shape[2], csp_proj_te.shape[3]))

    csp_tr = get_csp_feature(csp_proj_tr)
    csp_te = get_csp_feature(csp_proj_te)

    return csp_tr, csp_te


def SVM_classify(X_tr, X_te, Y_tr, Y_te):

    # 初始化SVM分类器
    svm = SVC()
    # 分类器训练
    svm.fit(X_tr, Y_tr)
    # 通过训练好的SVM得到预测值
    y_pred = svm.predict(X_te)

    # 将预测值与标签进行对比计算出分数
    acc = accuracy_score(Y_te, y_pred)
    print('识别准确率:', acc)

    labels = list(set(Y_tr))
    # 画混淆矩阵
    CM = confusion_matrix(Y_te, y_pred, labels=labels)
    print('混淆矩阵:\n', CM)
    for i in range(len(labels)):
        print(f'class {labels[i]}:', CM[i, i] / np.sum(CM[i]))

    return acc, CM


if __name__ == '__main__':
    X, y = data_preprocess('S2_LinxinXu_output_2025-6-5_10_42_27.json')
    # # 读取文件（180个试次(共三类)，11个电极通道，每个试次3s数据，250Hz采样，1-40Hz带通滤波）
    # X = np.load(current_directory+'/data/data.npy')
    # y = np.load(current_directory+'/data/labels.npy')

    acc_ran = np.zeros(k_ran)  # 不同随机数的相应结果
    CM_ran = np.zeros((n_classes, n_classes))  # confusion matrix with different rans
    for rann in range(k_ran):
        # train-test separation
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=rann, stratify=y)
        # OVRFBCSP得到CSP投影信号
        X_train, X_test = OVRFBCSP(X_train, X_test, Y_train)
        # 得到CSP方差特征再用SVM分类识别
        acc, CM = SVM_classify(X_train, X_test, Y_train, Y_test)

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
    for ii in range(n_classes):
        print(CM_ran[ii, ii] / np.sum(CM_ran[ii]))
    CM_ran_acc = np.zeros((n_classes, n_classes))
    for cla in range(n_classes):
        CM_ran_acc[cla] = CM_ran[cla] / np.sum(CM_ran[cla])
    sns.set()
    f, ax = plt.subplots()
    sns.heatmap(CM_ran_acc, annot=True, ax=ax)  # 画热力图

    ax.set_title('confusion matrix, acc=' + str(acc_ran_avg))  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.show()