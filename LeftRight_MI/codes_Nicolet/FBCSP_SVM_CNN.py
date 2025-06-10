import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

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


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = fs * 0.5  # 奈奎斯特采样频率
    low, high = lowcut / nyq, highcut / nyq
    # # # 滤波器的二阶截面表示
    sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
    # filtering
    y = signal.sosfiltfilt(sos, data)

    return y


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
    filt = 10  # 滤波器组的个数（8Hz-30Hz，间隔4Hz，重叠2Hz）
    features = 4  # 两类数据，每类数据2个特征，所以一个频带4个特征

    csp_proj_tr = np.zeros((tr_trials, filt*features, points))
    csp_proj_te = np.zeros((te_trials, filt*features, points))

    # 用滤波器组对epoch数据进行滤波并得到CSP投影信号
    for freq in range(filt):
        f_low = 8 + freq * 2  # low frequency
        f_high = 12 + freq * 2  # high frequency
        epochs_filt_tr = butter_bandpass_filter(X_tr, f_low, f_high, fs)
        epochs_filt_te = butter_bandpass_filter(X_te, f_low, f_high, fs)
        # 得到CSP投影信号，csp_porj_tr：试次数目*（10个滤波器*4个特征）*时间点数
        csp_proj_tr[:, freq*features:(freq+1)*features], csp_proj_te[:, freq*features:(freq+1)*features] =\
            csp_projected(epochs_filt_tr, epochs_filt_te, Y_tr)

    return csp_proj_tr, csp_proj_te


def SVM_classify(X_tr, X_te, Y_tr, Y_te):
    csp_tr, csp_te = csp_feature(X_tr, X_te)

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
    print('混淆矩阵:\n', CM)
    print('左手MI:', CM[0, 0] / np.sum(CM[0]), ' 右手MI:', CM[1, 1] / np.sum(CM[1]))

    return acc, CM


def EEGNet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5, kernLength=64, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
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
    # start the model
    input_main = Input((Chans, Samples, 1))
    block1 = Conv2D(40, (1, 13),
                    input_shape=(Chans, Samples, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(40, (Chans, 1), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


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

    ################################ Model ################################
    # kernLength, F1, D, F2 = 32, 8, 2, 16
    # model = EEGNet(nb_classes=2, Chans=chans, Samples=samples,
    #                dropoutRate=0.5, kernLength=kernLength, F1=F1, D=D, F2=F2,
    #                dropoutType='Dropout')

    model = ShallowConvNet(nb_classes=2, Chans=chans, Samples=samples, dropoutRate=0.5)

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

    history = model.fit(X_train, Y_train, batch_size=5, epochs=100,
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

    ################################ Testing ################################
    # model1 = EEGNet(nb_classes=2, Chans=chans, Samples=samples,
    #                 dropoutRate=0.5, kernLength=kernLength, F1=F1, D=D, F2=F2,
    #                 dropoutType='Dropout')
    model1 = ShallowConvNet(nb_classes=2, Chans=chans, Samples=samples, dropoutRate=0.5)

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
    print('混淆矩阵:\n', CM)
    print('左手MI:', CM[0, 0] / np.sum(CM[0]), ' 右手MI:', CM[1, 1] / np.sum(CM[1]))

    return acc, CM


if __name__ == '__main__':
    # 读取文件（80个试次(共两类)，21个电极通道，每个试次2.5s数据，500Hz采样，8-30Hz带通滤波）
    X = np.load('data.npy')
    y = np.load('labels.npy')

    X *= 1e5  # 使绝对值在1附近

    # 随机k_ran次
    k_ran = 10
    acc_ran = np.zeros(k_ran)  # 不同随机数的相应结果
    CM_ran = np.zeros((2, 2))  # confusion matrix with different rans
    for rann in range(k_ran):
        # train-test separation
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=rann, stratify=y)
        # FBCSP得到CSP投影信号
        # X_train, X_test = FBCSP(X_train, X_test, Y_train)
        # 得到CSP方差特征再用SVM分类识别
        # acc, CM = SVM_classify(X_train, X_test, Y_train, Y_test)
        # 通过CNN分类识别
        acc, CM = CNN_classify(X_train, X_test, Y_train, Y_test)

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