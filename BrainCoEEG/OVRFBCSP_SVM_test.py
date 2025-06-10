import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

import os
import yaml
import copy
import joblib

from MI_preprcessing import data_preprocess

current_directory = os.path.dirname(os.path.abspath(__file__))

with open(f'{current_directory}/settings.yaml', 'r') as f:
    settings_config = yaml.load(f, Loader=yaml.FullLoader)
fs = settings_config['fs']  # 采样率
n_classes = settings_config['n_classes']  # 类别个数
n_trials = settings_config['n_trials']  # 数据采集试次数量
n_runs = settings_config['n_runs']  # 数据采集轮次
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

 
def csp_projected(x_test, Filter):

    # 得到CSP投影信号，csp_porj_te：试次数目*10个滤波器*（n_classes*2个特征）*时间点数
    csp_projected_te = np.zeros((x_test.shape[0], n_components, x_test.shape[2]))
    for t, xx in enumerate(x_test):
        # projecting the data onto the CSP filters
        csp_projected_te[t] = np.dot(Filter, xx)

    return csp_projected_te


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


def OVRFBCSP(X_te, csp_filters):
    te_trials = X_te.shape[0]  # 测试数据包含的试次数量
    points = X_te.shape[2]  # 每个trial的点数

    X_filt_te = FB_filter(X_te)

    csp_proj_te = np.zeros((te_trials, n_filters, n_classes*n_components, points))

    for filt in range(n_filters):
        epochs_filt_te = X_filt_te[:, filt, :, :].copy()

        for ovr in range(n_classes):
            csp_filter = csp_filters[(filt * n_classes) + ovr]
            csp_projected_te = csp_projected(epochs_filt_te, csp_filter)

            csp_proj_te[:, filt, ovr * n_components:(ovr + 1) * n_components, :] = csp_projected_te

    csp_proj_te = np.reshape(csp_proj_te, (csp_proj_te.shape[0], csp_proj_te.shape[1] * csp_proj_te.shape[2], csp_proj_te.shape[3]))

    csp_te = get_csp_feature(csp_proj_te)

    return csp_te


def SVM_classify(X_te, Y_te, svm):

    # 通过训练好的SVM得到预测值
    y_pred = svm.predict(X_te)

    # 将预测值与标签进行对比计算出分数
    acc = accuracy_score(Y_te, y_pred)
    print('识别准确率:', acc)

    labels = list(set(Y_te))
    # 画混淆矩阵
    CM = confusion_matrix(Y_te, y_pred, labels=labels)
    print('混淆矩阵:\n', CM)
    for i in range(len(labels)):
        print(f'class {labels[i]}:', CM[i, i] / np.sum(CM[i]))

    return acc, CM


if __name__ == '__main__':
    X, y = data_preprocess('S2_LinxinXu_output_2025-6-5_10_42_27.json')
    # # 读取文件（180个试次(共三类)，6轮，11个电极通道，每个试次3s数据，250Hz采样，1-40Hz带通滤波）

    X_test = copy.deepcopy(X[n_trials-int(n_trials/n_runs):])
    Y_test = copy.deepcopy(y[n_trials-int(n_trials/n_runs):])

    # 读取模型参数
    clf = joblib.load('model.m')
    CSP_filters, SVM_model = clf[0], clf[1]

    # OVRFBCSP得到CSP投影信号
    X_test = OVRFBCSP(X_test, CSP_filters)
    # 得到CSP方差特征再用SVM分类识别
    acc, CM = SVM_classify(X_test, Y_test, SVM_model)

    sns.set()
    f, ax = plt.subplots()
    sns.heatmap(CM, annot=True, ax=ax)  # 画热力图
    ax.set_title('confusion matrix, acc=' + str(acc))  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.show()