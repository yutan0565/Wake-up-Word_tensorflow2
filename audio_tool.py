import librosa
import numpy as np

import librosa.display
from configuration import Config



def mel_spectrogram_process(signal, sr):
    # 정규화 만들기
    signal = signal / max(np.abs(signal))
    # 오디오 데이터에 대해서 window size만큼 stft 해주기, 그리고 모두 양수로 만들어 뒤집어 주기
    S = librosa.core.stft(signal,
                          n_fft=Config.n_fft,
                          hop_length=Config.hop_length,
                          win_length=Config.win_length)
    D = np.abs(S) ** 2  # 모두 양수로 날려주기
    # n_mels 필터의 모양을 나타낸다.
    mel_basis = librosa.filters.mel(sr,
                                    Config.n_fft,
                                    n_mels=Config.n_mels)
    mel_S = np.dot(mel_basis, D)
    # 마지막에는 log로 반환 해좌야 scale이 맞는다
    log_S = librosa.power_to_db(mel_S, ref=np.max)
    return log_S


def mfcc_process(signal, sr):
    # 보통 coefficent에서 13까지만 사용함
    log_mel_spectrogram = mel_spectrogram_process(signal, sr)
    mfcc = librosa.feature.mfcc(S=log_mel_spectrogram, n_mfcc=Config.num_mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    return delta2_mfcc

def spec_regularization(spectrogram):
  revverse_spectrogram = -spectrogram
  cut_spectrogram = revverse_spectrogram[:][:-50]
  zero_spectrogram = np.where( cut_spectrogram > 60 , 0, cut_spectrogram)
  min_value = np.min(zero_spectrogram)
  max_value = np.max(zero_spectrogram)
  min_max_scale_spectrogram =  (zero_spectrogram - min_value) / (max_value - min_value)
  return min_max_scale_spectrogram