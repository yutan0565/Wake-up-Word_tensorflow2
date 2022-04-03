import pyaudio
import librosa
import numpy as np
import librosa.display
from configuration import Config
import soundfile as sf
# import tensorflow as tf

import tflite_runtime.interpreter as tflite

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

import sys
import time
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

audio = pyaudio.PyAudio()

# for index in range(audio.get_device_count()):
#     desc = audio.get_device_info_by_index(index)
#     print("DEVICE: {device}, INDEX: {index}, RATE: {rate} ".format(
#         device=desc["name"], index=index, rate=int(desc["defaultSampleRate"])))

CHUNK = Config.sample_cut
RATE = 30000

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True,
                frames_per_buffer=CHUNK, input_device_index=1)
ffts = []
stfts = []
f_ffts = []
log_specs = []
MFCCs_list = []
def get_librosa_mfcc(signal, sr):
    # MFCC

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

tflite_model_path = Config.tflite_file_path
interpreter = tflite.Interpreter(tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print("프로그램 시작!!")

detection_image_path = Config.base_path +"show_image/wuw_detection.jpg"
image_pil = Image.open(detection_image_path)
detection_image = np.array(image_pil)

def close_figure(event):
    if event.key == ' ' or event.key == 'a':
        plt.close(event.canvas.figure)

def plot_time_series(data, title):
    fig = plt.figure(figsize=(7, 4))
    plt.title(title+'  wave')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 5, len(data)), data)
    # plt.show()

while (True):
    data = np.fromstring(stream.read(CHUNK), dtype=np.float32)
    #print(int(np.average(np.abs(data))))
    mfccs = get_librosa_mfcc(data,Config.sample_rate)
    input_tensor = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1] ,1)
    # print(input_tensor.shape)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    val = output_data[0]
    print("{:.03f}".format(val[3]))
    # print(len(data))
    sf.write("./test.wav", data, RATE)
    # if val > Config.thres_hold:
    if val[3] > Config.thres_hold:
        print("감지!!")
        print(len(data))
        sf.write("./test.wav", data, RATE)
        fig = plt.figure()
        plt.imshow(mfccs, cmap='inferno', origin='lower')
        # plt.imshow(detection_image)
        # plot_time_series(data, "short")
        plt.gcf().canvas.mpl_connect('key_press_event', close_figure)
        plt.show()

    else:
        print("없음")
    print(time.time())
    time.sleep(1)


stream.stop_stream()
stream.close()
p.terminate()


