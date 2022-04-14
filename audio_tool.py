import librosa
import numpy as np

import librosa.display
from configuration import Config

def max_scaler(signal):
    sig = signal / max(max(signal), max(np.abs(signal)))
    return sig

def mel_spectrogram_process(signal, sr):
    """
    :param signal: 음성 신호에 대한, data 값
    :param sr: sampleing rate
    :return: Mel-log Spectrogram
    """
    signal = signal / max(np.abs(signal))
    S = librosa.core.stft(signal,
                          n_fft=Config.n_fft,
                          hop_length=Config.hop_length,
                          win_length=Config.win_length)
    D = np.abs(S) ** 2
    mel_basis = librosa.filters.mel(sr,
                                    Config.n_fft,
                                    n_mels=Config.n_mels)
    mel_S = np.dot(mel_basis, D)
    log_S = librosa.power_to_db(mel_S, ref=np.max)
    return log_S


def mfcc_process(signal, sr):
    # 보통 coefficent에서 13까지만 사용함
    log_mel_spectrogram = mel_spectrogram_process(signal, sr)
    mfcc = librosa.feature.mfcc(S=log_mel_spectrogram, n_mfcc=Config.num_mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    return delta2_mfcc

def spec_regularization(spectrogram):
    """
    :param spectrogram: Mel-log Spectrogram의 결과값
   :return: Crop, 양수화 , min-max scale 변환 적용 결과 값
    """
    revverse_spectrogram = -spectrogram
    cut_spectrogram = revverse_spectrogram[:][:-50]
    zero_spectrogram = np.where( cut_spectrogram > 60 , 0, cut_spectrogram)
    min_value = np.min(zero_spectrogram)
    max_value = np.max(zero_spectrogram)
    min_max_scale_spectrogram =  (zero_spectrogram - min_value) / (max_value - min_value)
    return min_max_scale_spectrogram