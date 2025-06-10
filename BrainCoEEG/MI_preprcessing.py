# -*- coding: utf-8 -*-
"""
Created on Fri May 23 13:39:25 2025

@author: xingl
"""

import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import butter, iirnotch, sosfilt, filtfilt
import matplotlib.pyplot as plt
import yaml
import os

current_directory = os.path.dirname(os.path.abspath(__file__))

with open(f'{current_directory}/settings.yaml', 'r') as f:
    settings_config = yaml.load(f, Loader=yaml.FullLoader)
fs = settings_config['fs']  # 采样率
time_length = settings_config['time_length']  # 一组数据时长

channel_names = ['P8','P7','T8','T7','F8','F7','O2','O1','P4','P3','C4','C3','F4','F3','FP2','FP1',
                 'TP10','TP9','FT10','FT9','CP6','CP5','FC6','FC5','CP2','CP1','FC2','FC1','IO','Pz','Cz','Fz']

channel_names_31 = ['P8','P7','T8','T7','F8','F7','O2','O1','P4','P3','C4','C3','F4','F3','FP2','FP1',
                    'TP10','TP9','FT10','FT9','CP6','CP5','FC6','FC5','CP2','CP1','FC2','FC1','Pz','Cz','Fz']

channel_names_MI = ['C4', 'C3', 'CP6', 'CP5', 'FC6', 'FC5', 'CP2', 'CP1', 'FC2', 'FC1', 'Cz']

channel_names_new = ['Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6',
                     'Cz', 'C3', 'C4', 'T7', 'T8',
                     'Pz', 'P3', 'P4', 'P7', 'P8', 'CP1', 'CP2', 'CP5', 'CP6']


def data_extract(filepath):
    data = pd.read_json(filepath)
    # Print all column names
    # print(data.columns.tolist())

    eeg_data = data['eegRes'].iloc[2:-2]
    labels = data['label'].iloc[2:-2]

    return eeg_data, labels


def assign_channel_name(signal, channel_name=channel_names):
    eeg_df = pd.DataFrame(signal, columns=channel_name)
    eeg_df_31 = eeg_df.drop(columns="IO")  # remove IO channel, invalid channel
    eeg_df_31 = np.array(eeg_df_31)
    return eeg_df_31


def slicing_filtering(signal, f0=50, Q=30, lowcut=1, highcut=40, order=4):

    # data slice
    signal_sliced = signal[:int(time_length * fs)]

    # data filter
    # 1. remove DC offset
    signal_detrend = signal_sliced - np.mean(signal_sliced, axis=0)
    # 2. notch filter
    b, a = iirnotch(f0, Q, fs)
    # Apply notch filter
    notch_filtered = filtfilt(b, a, signal_detrend, axis=0)
    # 3. bandpass filter
    nyquist = 0.5*fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b2, a2 = butter(order, [low, high], btype = 'band')
    bp_filtered = filtfilt(b2, a2, notch_filtered)

    return bp_filtered


def plot_eeg(eeg_data, channel_names=channel_names_31, fs=250, title='EEG Plot'):

    if eeg_data.shape[1] != len(channel_names):
        raise ValueError("Length of channel_names must match number of channels in eeg_data")

    time = np.arange(eeg_data.shape[0]) / fs  # Time axis in seconds
    offset = 100  # Vertical offset for each channel (adjust based on data scale)

    plt.figure(figsize=(12, 10))
    for i in range(eeg_data.shape[1]):
        plt.plot(time, eeg_data[:, i] + i * offset, label=channel_names[i])

    plt.yticks(np.arange(len(channel_names)) * offset, channel_names)
    plt.xlabel('Time (s)')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def data_preprocess(filename):
    file_dir = os.path.join(current_directory, "data")
    file_path = os.path.join(file_dir, filename)

    eeg_data, raw_labels = data_extract(file_path)

    all_preprocessed_data = np.zeros((180, int(time_length * fs), 31))
    all_labels = []

    for i in range(180):
        eeg_epoch = eeg_data.iloc[i]
        eeg_epoch = assign_channel_name(eeg_epoch)
        filtered_eeg = slicing_filtering(eeg_epoch)
        all_preprocessed_data[i] = filtered_eeg

        label = 0
        if raw_labels.iloc[i] == 'left':
            label = 1
        elif raw_labels.iloc[i] == 'right':
            label = 2
        all_labels.append(label)

    # plot example
    # plot_eeg(all_preprocessed_data[0])

    channel_index = []
    for chan_MI in channel_names_new:
        channel_index.append(channel_names_31.index(chan_MI))

    data = np.transpose(all_preprocessed_data[:, :, channel_index], (0, 2, 1))
    labels = np.array(all_labels)

    # # save data
    # np.save(os.path.join(file_dir, "data.npy"), data)
    # np.save(os.path.join(file_dir, "labels.npy"), labels)

    return data, labels


if __name__ == "__main__":
    filename = 'xl_output_2025-5-23_10_57_0.json'
    data_preprocess(filename)