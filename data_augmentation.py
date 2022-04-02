import soundfile as sf
import os
from os import listdir
from os.path import isdir, join
import librosa
import numpy as np
import shutil
from distutils.dir_util import copy_tree
import matplotlib.pyplot as plt

from configuration import Config

def plot_time_series(data, title):
    fig = plt.figure(figsize=(7, 4))
    plt.title(title+'  wave')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 5, len(data)), data)
    plt.show()

# Whit Noise
# 기존 소리에 잡음을 넣어줌
def adding_white_noise(data, end_path,  count, sr=16000, noise_rate=0.001):
    # noise 방식으로 일반적으로 쓰는 잡음 끼게 하는 겁니다.
    sig, sr = librosa.load(data, sr=16000)
    wn = np.random.randn(len(sig))
    data_wn = sig + noise_rate*wn


    sf.write( end_path, data_wn, sr)
    return data_wn

#stretch_sound
# 테이프 늘어진 것처럼 들린다.
def stretch_sound(data, end_path,  count, sr=16000, rate=0.8):
    sig, sr = librosa.load(data, sr=16000)
    stretch_data = librosa.effects.time_stretch(sig, rate)



    sf.write( end_path, stretch_data, sr)
    return stretch_data

# minus_sound
# x 축 기준으로 뒤집기 (사람에게는 똑같이 들림)
def minus_sound(data, end_path,  count, sr=16000):
    sig, sr = librosa.load(data, sr=16000)
    temp_numpy = (-1)*sig

    sf.write( end_path, temp_numpy, sr)
    return temp_numpy

for user in Config.user_list:
    print(user+ "start augmentation")
    for index, type in enumerate(Config.target_list):
      # 데이터 보내줄 곳
      start_path = '/'.join([Config.aug_dataset_path,user, type])

      # if not os.path.exists(start_path):
      #           os.makedirs(start_path)

      all_file = listdir(start_path)
      count = 1
      aug_cut = len(all_file) * Config.aug_rate
      for file_name in all_file:
        if count > aug_cut:
          break
        file_path =  start_path +"/"+file_name
        path = Config.base_path + Config.dataset_type + '/' + user + '/' + type + '/'

        noise_name =  'noise_' + user + '_' + '{0:04d}'.format(count) + '_' + type + '.wav'
        noise_end_path = path + noise_name
        adding_white_noise(file_path, noise_end_path, count)

        stretch_name =  'stretch_' + user + '_' + '{0:04d}'.format(count) + '_' + type + '.wav'
        stretch_end_path = path + stretch_name
        stretch_sound(file_path, stretch_end_path,  count)

        minus_name = 'minus_' + user + '_' + '{0:04d}'.format(count) + '_' + type + '.wav'
        minus_end_path = path + minus_name
        minus_sound(file_path, minus_end_path,  count)
        copy_tree(start_path, path )
        count += 1