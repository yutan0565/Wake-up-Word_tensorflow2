import pyaudio

from configuration import Config
import soundfile as sf
import audio_tool as tool

import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import sys
import time
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")



# for index in range(audio.get_device_count()):
#     desc = audio.get_device_info_by_index(index)
#     print("DEVICE: {device}, INDEX: {index}, RATE: {rate} ".format(
#         device=desc["name"], index=index, rate=int(desc["defaultSampleRate"])))

CHUNK = int(Config.sample_cut / Config.stride_rate) # 8000




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


audio = pyaudio.PyAudio()
p = pyaudio.PyAudio()

stream_wuw = p.open(format=pyaudio.paFloat32, channels=1, rate=Config.sample_rate, input=True,
                frames_per_buffer=CHUNK, input_device_index=1)

stream_low = p.open(format=pyaudio.paFloat32, channels=1, rate=Config.low_power_sample_rate, input=True,
                frames_per_buffer=CHUNK, input_device_index=1)


print("프로그램 시작!!")
frame = []
initiate_frame = [0 for i in range(CHUNK)]
frame.append(initiate_frame)

while (True):
    temp_data = np.fromstring(stream_wuw.read(CHUNK), dtype=np.float32)

    # 기동어 감지 모델을 돌리고 있지 않는 상태
    if np.mean(temp_data) < Config.thres_hold_low_power:
        if np.array(frame).shape[0] == Config.stride_rate:
            frame.pop(0)
            frame.append(temp_data)
            data = frame[0]
            for i in range(1, Config.stride_rate):
                data = np.concatenate((data, frame[i]), axis=0)
        else:
            frame.append(temp_data)
            print(np.array(frame).shape)
        print("x")  # 기동어 인식 안하고 있는 상태
        continue

    if np.array(frame).shape[0] == Config.stride_rate:
        frame.pop(0)
        frame.append(temp_data)
        data= frame[0]
        for i in range(1, Config.stride_rate):
          data = np.concatenate( (data, frame[i]), axis = 0  )

        # 데이터 전처리 과정
        spectrogram = tool.mel_spectrogram_process(data, Config.sample_rate)
        regul_spectrogram = tool.spec_regularization(spectrogram)
        input_tensor = regul_spectrogram.reshape(1, regul_spectrogram.shape[0], regul_spectrogram.shape[1] ,1)
        # print(input_tensor.shape)

        # detection input 설정
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        val = output_data[0]
        # print("탄탄 : {:.03f}  병현 : {:.03f}  정률 : {:.03f}  성우 : {:.03f}  유탄 : {:.03f}  no : {:.03f}".format(val[0],val[1],val[2],val[3], val[4], val[5]))
        # print(len(data))

        sf.write("./test.wav", data, Config.sample_rate)

        # if val > Config.thres_hold:
        if val[4] > Config.thres_hold:
            print("감지!!")
            # print(len(data))
            sf.write("./test.wav", data, Config.sample_rate)
            fig = plt.figure()
            #plt.imshow(mfccs, cmap='inferno', origin='lower')
            plt.imshow(detection_image)
            # plot_time_series(data, "short")
            plt.gcf().canvas.mpl_connect('key_press_event', close_figure)
            plt.show()
            frame = []

            interpreter_recog.set_tensor(input_details_recog[0]['index'], input_tensor)
            interpreter_recog.invoke()
            output_data_recog = interpreter_recog.get_tensor(output_details_recog[0]['index'])

            val_recog = output_data_recog[0]
            print("User_01 : {:.03f}  User_02 : {:.03f}  User_03 : {:.03f}".format(val_recog[0], val_recog[1], val_recog[2]))

            if val_recog[0] > Config.thres_hold_recog :
                print("검증 성공 : 서버와 연결을 시작 합니다.")
                # print(len(data))
                fig = plt.figure()
                plt.imshow(detection_image)
                # plot_time_series(data, "short")
                plt.gcf().canvas.mpl_connect('key_press_event', close_figure)
                plt.show()
            else:
                print("검증 실패 : 사용자 등록을 해주세요")

        else:
            print("0")  # 기동어 들을 준비 완료


    else:
        frame.append(temp_data)
        print(np.array(frame).shape)


stream_wuw.stop_stream()
stream_wuw.close()
p.terminate()

"""
while (True):
  temp_data = np.fromstring(stream_wuw.read(CHUNK), dtype=np.float32)
  if np.array(frame).shape[0] == Config.stride_rate:
    frame.pop(0)
    frame.append(temp_data)
    data= frame[0]
    for i in range(1, Config.stride_rate):
      data = np.concatenate( (data, frame[i]), axis = 0  )

    spectrogram = tool.mel_spectrogram_process(data, Config.sample_rate)
    regul_spectrogram = tool.spec_regularization(spectrogram)


    input_tensor = regul_spectrogram.reshape(1, regul_spectrogram.shape[0], regul_spectrogram.shape[1] ,1)
    # print(input_tensor.shape)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    val = output_data[0]
    # print("탄탄 : {:.03f}  병현 : {:.03f}  정률 : {:.03f}  성우 : {:.03f}  유탄 : {:.03f}  no : {:.03f}".format(val[0],val[1],val[2],val[3], val[4], val[5]))
    # print(len(data))
    sf.write("./test.wav", data, Config.sample_rate)
    # if val > Config.thres_hold:
    if val[4] > Config.thres_hold:
        print("감지!!")
        # print(len(data))
        sf.write("./test.wav", data, Config.sample_rate)
        fig = plt.figure()
        #plt.imshow(mfccs, cmap='inferno', origin='lower')
        plt.imshow(detection_image)
        # plot_time_series(data, "short")
        plt.gcf().canvas.mpl_connect('key_press_event', close_figure)
        plt.show()
        frame = []

        interpreter_recog.set_tensor(input_details_recog[0]['index'], input_tensor)
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
