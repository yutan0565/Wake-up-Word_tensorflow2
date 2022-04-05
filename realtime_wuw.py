import pyaudio

from configuration import Config
import soundfile as sf
import MFCC_maker

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

CHUNK = int(Config.sample_cut / Config.stride_rate)
RATE = 32000

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True,
                frames_per_buffer=CHUNK, input_device_index=1)


tflite_model_path = Config.tflite_file_path
interpreter = tflite.Interpreter(tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


tflite_model_path_recog = Config.tflite_file_path_recog
interpreter_recog = tflite.Interpreter(tflite_model_path_recog)
interpreter_recog.allocate_tensors()
input_details_recog = interpreter_recog.get_input_details()
output_details_recog = interpreter_recog.get_output_details()


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

frame = []
while (True):
  temp_data = np.fromstring(stream.read(CHUNK), dtype=np.float32)
  if np.array(frame).shape[0] == Config.stride_rate:
    frame.pop(0)
    frame.append(temp_data)
    data= frame[0]
    for i in range(1, Config.stride_rate):
      data = np.concatenate( (data, frame[i]), axis = 0  )

    #mfccs = MFCC_maker.mfcc_process(data, Config.sample_rate)
    mfccs = MFCC_maker.mel_spectrogram_process(data, Config.sample_rate)
    input_tensor = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1] ,1)
    # print(input_tensor.shape)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    val = output_data[0]
    # print("탄탄 : {:.03f}  병현 : {:.03f}  정률 : {:.03f}  성우 : {:.03f}  유탄 : {:.03f}  no : {:.03f}".format(val[0],val[1],val[2],val[3], val[4], val[5]))
    # print(len(data))
    sf.write("./test.wav", data, RATE)
    # if val > Config.thres_hold:
    if val[4] > Config.thres_hold:
        print("감지!!")
        # print(len(data))
        sf.write("./test.wav", data, RATE)
        fig = plt.figure()
        #plt.imshow(mfccs, cmap='inferno', origin='lower')
        plt.imshow(detection_image)
        # plot_time_series(data, "short")
        plt.gcf().canvas.mpl_connect('key_press_event', close_figure)
        plt.show()
        frame = []

        interpreter_recog.set_tensor(input_details[0]['index'], input_tensor)
        interpreter_recog.invoke()
        output_data_recog = interpreter_recog.get_tensor(output_details_recog[0]['index'])
        val_recog = output_data_recog[0]
        print("User_01 : {:.03f}  User_02 : {:.03f}  User_03 : {:.03f}".format(val_recog[0], val_recog[1], val_recog[2]))
        if val_recog[0] > Config.thres_hold_recog :
            print("검증 성공")
            # print(len(data))
            fig = plt.figure()
            plt.imshow(detection_image)
            # plot_time_series(data, "short")
            plt.gcf().canvas.mpl_connect('key_press_event', close_figure)
            plt.show()
        else:
            print("검증 실패")

    else:
        print("없음")
    # print(time.time())

  else:
    frame.append(temp_data)
    print(np.array(frame).shape)


stream.stop_stream()
stream.close()
p.terminate()

"""
import pyaudio
import librosa
import numpy as np
import librosa.display
from configuration import Config
import soundfile as sf
import MFCC_maker

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
RATE = 32000

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True,
                frames_per_buffer=CHUNK, input_device_index=1)


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
    mfccs = MFCC_maker.mfcc_process(data, Config.sample_rate)
    input_tensor = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1] ,1)
    # print(input_tensor.shape)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    val = output_data[0]
    print("탄탄 : {:.03f}  병현 : {:.03f}  정률 : {:.03f}  유탄 : {:.03f}  no : {:.03f}".format(val[0],val[1],val[2],val[3], val[4]))
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
"""

