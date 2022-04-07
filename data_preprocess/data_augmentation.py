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
def adding_white_noise(data, end_path,  noise_rate=0.002):
    # noise 방식으로 일반적으로 쓰는 잡음 끼게 하는 겁니다.
    signal, sr = librosa.load(data, sr=Config.sample_rate)
    sig = signal[int(-Config.sample_cut - Config.click): int(-Config.click)]
    wn = np.random.randn(len(sig))
    data_wn = sig + noise_rate*wn
    sf.write( end_path, data_wn, sr)
    return data_wn

#stretch_sound
# 테이프 늘어진 것처럼 들린다.
def stretch_sound(file_path, end_path,  rate=0.6):
    signal, sr = librosa.load(file_path, sr=Config.sample_rate)
    sig = signal[int(-Config.sample_cut - Config.click): int(-Config.click)]
    stretch_data = librosa.effects.time_stretch(sig, rate)
    sf.write( end_path, stretch_data, sr)
    return stretch_data

# minus_sound
# x 축 기준으로 뒤집기 (사람에게는 똑같이 들림)
def minus_sound(file_path, end_path):
    signal, sr = librosa.load(file_path, sr=Config.sample_rate)
    sig = signal[int(-Config.sample_cut - Config.click): int(-Config.click)]
    temp_numpy = (-1)*sig
    sf.write( end_path, temp_numpy, sr)
    return temp_numpy

def pitch_sound(file_path, end_path, pitch_factor=5):   # end_path, count,
    signal, sr = librosa.load(file_path, sr=Config.sample_rate)
    sig = signal[int(-Config.sample_cut - Config.click): int(-Config.click)]
    pitch_data = librosa.effects.pitch_shift(sig, sr, pitch_factor)
    sf.write( end_path, pitch_data, sr)
    return pitch_data


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

        # 기본 노이즈 추가
        noise_name =  'noise_' + user + '_' + '{0:04d}'.format(count) + '_' + type + '.wav'
        noise_end_path = path + noise_name
        adding_white_noise(file_path, noise_end_path)

        # 말 늘이기
        stretch_name =  'stretch_' + user + '_' + '{0:04d}'.format(count) + '_' + type + '.wav'
        stretch_end_path = path + stretch_name
        stretch_sound(file_path, stretch_end_path)

        # 단순 뒤집기
        minus_name = 'minus_' + user + '_' + '{0:04d}'.format(count) + '_' + type + '.wav'
        minus_end_path = path + minus_name
        minus_sound(file_path, minus_end_path)

        # 음 높이기
        pitch_name = 'pitch_' + user + '_' + '{0:04d}'.format(count) + '_' + type + '.wav'
        pitch_end_path = path + pitch_name
        pitch_sound(file_path, pitch_end_path)

        #shift 해주기
        for


        # 원본도 같이 옮기기
        copy_tree(start_path, path )
        count += 1