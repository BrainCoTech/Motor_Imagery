import asyncio
import numpy as np

import bc_device_sdk as sdk
from bc_device_sdk import MessageParser, MsgType, NoiseTypes
from eeg_cap_model import (
    EEGData,
    IMUData,
    eeg_cap,
    get_addr_port,
    set_env_noise_cfg,
    set_eeg_buffer_length,
)
from MI_preprcessing import slicing_filtering
from OVRFBCSP_SVM_test import OVRFBCSP

import joblib
import os
import yaml

current_directory = os.path.dirname(os.path.abspath(__file__))

with open(f'{current_directory}/settings.yaml', 'r') as f:
    settings_config = yaml.load(f, Loader=yaml.FullLoader)
fs = settings_config['fs']  # 采样率
time_length = settings_config['time_length']  # 一组数据时长
n_classes = settings_config['n_classes']  # 类别个数
n_filters = settings_config['n_filters']  # 滤波器组的个数（8Hz-30Hz，间隔4Hz，重叠2Hz）
n_components = settings_config['n_components']  # CSP特征选取个数

# EEG数据
num_channels = 32  # 通道数
eeg_buffer_length = 1000  # 默认缓冲区长度, 1000个数据点，每个数据点有32个通道，每个通道的值类型为float32，即4字节，大约占用128KB内存, 1000 * 32 * 4 = 128000 bytes
eeg_seq_num = None  # EEG数据包序号
eeg_values = np.zeros((num_channels, eeg_buffer_length))  # 32通道的EEG数据

CH_LABELS = ["P8", "P7", "T8", "T7", "F8", "F7", "O2", "O1", "P4", "P3",
             "C4", "C3", "F4", "F3", "FP2", "FP1", "TP10", "TP9", "FT10", "FT9",
             "CP6", "CP5", "FC6", "FC5", "CP2", "CP1", "FC2", "FC1", "IO", "Pz",
             "Cz", "Fz"]
CH_SELECT = ['Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6',
             'Cz', 'C3', 'C4', 'T7', 'T8',
             'Pz', 'P3', 'P4', 'P7', 'P8', 'CP1', 'CP2', 'CP5', 'CP6']
# GND:AFZ; REF:FCZ
CH_ID = []
for ch in CH_SELECT:
    CH_ID.append(CH_LABELS.index(ch))

# 读取模型参数
clf = joblib.load('model.m')
CSP_filters, SVM_model = clf[0], clf[1]


def print_imu_data():
    fetch_num = 100  # 每次获取的数据点数, 超过缓冲区长度时，返回缓冲区中的所有数据
    clean = True  # 是否清空缓冲区
    imu_buff = eeg_cap.get_imu_buffer(fetch_num, clean)
    imu_result = []
    for i in range(len(imu_buff)):
        imu_result.append(IMUData.from_data(imu_buff[i]))

    result_str = "\n\t".join(map(str, imu_result))


def read_eeg_data():
    # 获取EEG数据
    fetch_num = 100  # 每次获取的数据点数, 超过缓冲区长度时，返回缓冲区中的所有数据
    clean = True  # 是否清空缓冲区
    eeg_buff = eeg_cap.get_eeg_buffer(fetch_num, clean)

    if len(eeg_buff) == 0:
        return

    eeg_data_arr = []
    for row in eeg_buff:
        eeg_data = EEGData.from_data(row)
        eeg_data_arr.append(eeg_data)

        # 检查数据包序号
        timestamp = eeg_data.timestamp
        global eeg_seq_num
        if eeg_seq_num is not None and timestamp != eeg_seq_num + 1:
            print(f"eeg_seq_num={eeg_seq_num}, timestamp={timestamp}")
        if eeg_seq_num is not None or timestamp == 2:  # 第一个数据包的时间戳有误
            eeg_seq_num = timestamp

        channel_values = eeg_data.channel_values
        # 更新每个通道的数据
        for channel in range(len(channel_values)):
            # fmt: off
            eeg_values[channel] = np.roll(eeg_values[channel], -1)  # 数据向左滚动，腾出最后一个位置
            eeg_values[channel, -1] = channel_values[channel]  # 更新最新的数据值

    eeg_data = np.array([slicing_filtering(eeg_values[CH_ID, -int(fs*time_length):])])

    return eeg_data


def MI_recognition(eeg_data):

    if eeg_data is None:
        return

    if np.max(np.abs(eeg_data)) == 0:
        return

    # OVRFBCSP得到CSP投影信号
    X_test = OVRFBCSP(eeg_data, CSP_filters)
    # 得到CSP方差特征再用SVM分类识别
    result = SVM_model.predict(X_test)

    return result


async def scan_and_connect():
    (addr, port) = await get_addr_port()
    client = eeg_cap.ECapClient(addr, port)

    # 连接设备，监听消息
    parser = MessageParser("eeg-cap-device", MsgType.EEGCap)
    await client.start_data_stream(parser)

    # 获取EEG配置
    msgId = await client.get_eeg_config()
    print(f"msgId: {msgId}")

    # 开始EEG数据流
    msgId = await client.start_eeg_stream()
    print(f"msgId: {msgId}")

    # 开始IMU数据流
    msgId = await client.start_imu_stream()
    print(f"msgId: {msgId}")


def init_cfg():
    print("Init cfg")
    set_env_noise_cfg(NoiseTypes.FIFTY, fs)  # 滤波器参数设置，去除50Hz电流干扰
    set_eeg_buffer_length(eeg_buffer_length)  # 设置EEG数据缓冲区长度
    sdk.set_msg_resp_callback(lambda msg: print(f"Message response: {msg}"))


async def main():
    init_cfg()
    await scan_and_connect()
    while True:
        eeg_data = read_eeg_data()
        result = MI_recognition(eeg_data)
        print(f"MI recognition result: {result}")
        # print_imu_data()
        await asyncio.sleep(0.1)  # 100ms


if __name__ == "__main__":
    asyncio.run(main())
