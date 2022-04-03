import librosa
import numpy as np
import librosa.display
from configuration import Config


# MFCC
ffts = []
stfts = []
f_ffts = []
log_specs = []
MFCCs_list = []

def get_librosa_mfcc(path):
    sig, sr = librosa.core.load(path, Config.sample_rate)
    signal = sig[-Config.sample_cut - Config.click:-Config.click]
    #     print('sr:', sr, ', sig shape:', sig.shape)
    #     print('length:', sig.shape[0]/float(sr), 'secs')
    #     plot_time_series(signal, "original")

    fft = np.fft.fft(signal)
    magnitude = np.abs(fft)
    frequency = np.linspace(0, Config.sample_rate, len(magnitude))
    f_ffts.append([frequency, magnitude])

    # calculate abs values on complex numbers to get magnitude
    spectrum = np.abs(fft)

    # half frequency variable
    f = np.linspace(0, Config.sample_rate, len(spectrum))
    left_spectrum = spectrum[:int(len(spectrum) / 2)]
    left_f = f[:int(len(spectrum) / 2)]
    ffts.append([left_f, left_spectrum])

    # Short-time FT
    stft = librosa.stft(signal, n_fft=Config.n_fft, hop_length=Config.hop_length)
    spectrogram = np.abs(stft)
    stfts.append([spectrogram, Config.sample_rate, Config.hop_length])
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    log_specs.append([log_spectrogram, Config.sample_rate, Config.hop_length])
    MFCCs = librosa.feature.mfcc(signal, Config.sample_rate,
                                 n_mfcc=Config.num_mfcc,
                                 n_fft=Config.n_fft,
                                 hop_length=Config.hop_length)
    MFCCs_list.append([MFCCs, Config.sample_rate, Config.hop_length])
    return MFCCs
