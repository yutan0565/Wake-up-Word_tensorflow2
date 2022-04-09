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

tflite_model_path_detect = Config.tflite_file_path
interpreter_detect = tflite.Interpreter(tflite_model_path_detect)
interpreter_detect.allocate_tensors()
input_details_detect = interpreter_detect.get_input_details()
output_details_detect = interpreter_detect.get_output_details()

tflite_model_path_recog = Config.tflite_file_path_recog_02
interpreter_recog = tflite.Interpreter(tflite_model_path_recog)
interpreter_recog.allocate_tensors()
input_details_recog = interpreter_recog.get_input_details()
output_details_recog = interpreter_recog.get_output_details()


detection_image_path = Config.base_path +"show_image/Detection.png"
detection_image_pil = Image.open(detection_image_path)
detection_image = np.array(detection_image_pil)

speaker_valid_image_path = Config.base_path +"show_image/Valid.png"
speaker_valid_image_pil = Image.open(speaker_valid_image_path)
speaker_valid_image = np.array(speaker_valid_image_pil)

speaker_invalid_image_path = Config.base_path +"show_image/Invalid.png"
speaker_invalid_image_pil = Image.open(speaker_invalid_image_path)
speaker_invalid_image = np.array(speaker_invalid_image_pil)




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


def make_ouput(input_tensor, interpreter, input_detail, output_detail):
    interpreter.set_tensor(input_detail[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_detail[0]['index'])
    return output_data[0]

def close_figure(event):
    if event.key == ' ' or event.key == 'a':
        plt.close(event.canvas.figure)

def show_result_image(result_image):
    fig = plt.figure()
    # plt.imshow(mfccs, cmap='inferno', origin='lower')
    plt.imshow(result_image)
    # plot_time_series(data, "short")
    plt.gcf().canvas.mpl_connect('key_press_event', close_figure)
    plt.show()

def make_frame(past_frame, input_data):
    past_frame.pop(0)
    past_frame.append(input_data)
    new_frame = past_frame
    new_data = past_frame[0]
    for i in range(1, Config.stride_rate):
        new_data = np.concatenate((new_data, past_frame[i]), axis=0)
    return new_frame, new_data

frame = []
initiate_frame = [0 for i in range(CHUNK)]
frame.append(initiate_frame)


print("프로그램 시작!!")
while (True):
    temp_data = np.fromstring(stream_wuw.read(CHUNK), dtype=np.float32)

    # 기동어 감지 모델을 돌리고 있지 않는 상태
    if np.mean(np.abs(temp_data)) < Config.thres_hold_low_power:
        if np.array(frame).shape[0] == Config.stride_rate:
            frame, _ = make_frame(frame, temp_data)
        else:
            frame.append(temp_data)
            print(np.array(frame).shape)
        print("xx")  # 기동어 인식 안하고 있는 상태
        continue

    if np.array(frame).shape[0] == Config.stride_rate:
        frame, signal = make_frame(frame, temp_data)

        spectrogram = tool.mel_spectrogram_process(signal, Config.sample_rate)
        regul_spectrogram = tool.spec_regularization(spectrogram)
        input_tensor = regul_spectrogram.reshape(1, regul_spectrogram.shape[0], regul_spectrogram.shape[1] ,1)

        val_detect = make_ouput( input_tensor ,interpreter_detect,input_details_detect, output_details_detect )

        for index_detect in Config.target_wake_word_index:
            if val_detect[index_detect] > Config.thres_hold_detect:
                print("hey_tantan : {:.02f}  hi_byeonghyeon : {:.02f}  hi_jeonglyul : {:.02f}".format(val_detect[0],
                                                                                                      val_detect[1],
                                                                                                      val_detect[2]))
                print("hi_sungwoo : {:.02f}  hi_yutan : {:.02f}  other : {:.02f}".format(val_detect[3],
                                                                                                      val_detect[4],
                                                                                                      val_detect[5]))
                print("{} 감지!!".format(Config.target_list[index_detect]))

                show_result_image(detection_image)
                val_recog = make_ouput(input_tensor, interpreter_recog, input_details_recog, output_details_recog)
                print("User_01 : {:.02f}  User_02 : {:.02f}  User_03 : {:.02f}  User_04 : {:.02f}".format(val_recog[0], val_recog[1], val_recog[2], val_recog[3]))
                flag= True
                for index_recog in Config.target_user_index:
                    if val_recog[index_recog] > Config.thres_hold_recog:
                        if flag:
                            print("검증 성공 : 안녕하세요. {} 님!!!".format(Config.user_list[index_recog]))
                            show_result_image(speaker_valid_image)
                        break
                    else:
                        if flag:
                            print("검증 실패 : 사용자 등록을 해주세요")
                            show_result_image(speaker_invalid_image)
                        flag = False
                frame = [] # frame 초기화
                break
            else:
                print("000000000")  # 기동어 들을 준비 완료
    else:
        frame.append(temp_data)
        print(np.array(frame).shape)


stream_wuw.stop_stream()
stream_wuw.close()
p.terminate()

"""
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

tflite_model_path_detect = Config.tflite_file_path
interpreter_detect = tflite.Interpreter(tflite_model_path_detect)
interpreter_detect.allocate_tensors()
input_details_detect = interpreter_detect.get_input_details()
output_details_detect = interpreter.get_output_details()

tflite_model_path_recog = Config.tflite_file_path_recog
interpreter_recog = tflite.Interpreter(tflite_model_path_recog)
interpreter_recog.allocate_tensors()
input_details_recog = interpreter_recog.get_input_details()
output_details_recog = interpreter_recog.get_output_details()


detection_image_path = Config.base_path +"show_image/Detection.png"
detection_image_pil = Image.open(detection_image_path)
detection_image = np.array(detection_image_pil)

speaker_valid_image_path = Config.base_path +"show_image/Valid.png"
speaker_valid_image_pil = Image.open(speaker_valid_image_path)
speaker_valid_image = np.array(speaker_valid_image_pil)

speaker_invalid_image_path = Config.base_path +"show_image/Invalid.png"
speaker_invalid_image_pil = Image.open(speaker_invalid_image_path)
speaker_invalid_image = np.array(speaker_invalid_image_pil)




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


def make_ouput(input_tensor, interpreter, input_detail, output_detail):
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

def close_figure(event):
    if event.key == ' ' or event.key == 'a':
        plt.close(event.canvas.figure)
        
def show_result_image(result_image):
    fig = plt.figure()
    # plt.imshow(mfccs, cmap='inferno', origin='lower')
    plt.imshow(result_image)
    # plot_time_series(data, "short")
    plt.gcf().canvas.mpl_connect('key_press_event', close_figure)
    plt.show()

while (True):
    temp_data = np.fromstring(stream_wuw.read(CHUNK), dtype=np.float32)

    # 기동어 감지 모델을 돌리고 있지 않는 상태
    if np.mean(np.abs(temp_data)) < Config.thres_hold_low_power:
        if np.array(frame).shape[0] == Config.stride_rate:
            frame.pop(0)
            frame.append(temp_data)
            data = frame[0]
            for i in range(1, Config.stride_rate):
                data = np.concatenate((data, frame[i]), axis=0)
        else:
            frame.append(temp_data)
            print(np.array(frame).shape)
        print("xx")  # 기동어 인식 안하고 있는 상태
        continue

    if np.array(frame).shape[0] == Config.stride_rate:
        frame.pop(0)
        frame.append(temp_data)
        data= frame[0]
        for i in range(1, Config.stride_rate):
          data = np.concatenate( (data, frame[i]), axis = 0  )

        # # 데이터 전처리 과정
        # spectrogram = tool.mel_spectrogram_process(data, Config.sample_rate)
        # regul_spectrogram = tool.spec_regularization(spectrogram)
        # input_tensor = regul_spectrogram.reshape(1, regul_spectrogram.shape[0], regul_spectrogram.shape[1] ,1)
    
        val_detect = make_ouput( input_tensor ,interpreter_detect,input_details_detect, output_details_detect )

        # print("탄탄 : {:.03f}  병현 : {:.03f}  정률 : {:.03f}  성우 : {:.03f}  유탄 : {:.03f}
        # no : {:.03f}".format(val_detect[0],val_detect[1],val_detect[2],val_detect[3], val_detect[4], val_detect[5]))
        # sf.write("./test.wav", data, Config.sample_rate)

        if val_detect[4] > Config.thres_hold:
            print("감지!!")
            show_result_image(detection_image)
            # print(len(data))
            # sf.write("./test.wav", data, Config.sample_rate)
            #fig = plt.figure()
            #plt.imshow(mfccs, cmap='inferno', origin='lower')
            #plt.imshow(detection_image)
            # plot_time_series(data, "short")
            #plt.gcf().canvas.mpl_connect('key_press_event', close_figure)
            #plt.show()
            frame = []

            # interpreter_recog.set_tensor(input_details_recog[0]['index'], input_tensor)
            # interpreter_recog.invoke()
            # output_data_recog = interpreter_recog.get_tensor(output_details_recog[0]['index'])
            #val_recog = output_data_recog[0]
            
            val_recog = make_ouput(input_tensor, interpreter_recog, input_details_recog, output_details_recog)
            
            print("User_01 : {:.03f}  User_02 : {:.03f}  User_03 : {:.03f}".format(val_recog[0], val_recog[1], val_recog[2]))

            if val_recog[0] > Config.thres_hold_recog :
                print("검증 성공 : 서버와 연결을 시작 합니다.")
                show_result_image(speaker_valid_image)
                # print(len(data))
                #fig = plt.figure()
                #plt.imshow(speaker_valid_image)
                # plot_time_series(data, "short")
                #plt.gcf().canvas.mpl_connect('key_press_event', close_figure)
                #plt.show()
            else:
                print("검증 실패 : 사용자 등록을 해주세요")
                show_result_image(speaker_invalid_image)
                #fig = plt.figure()
                #plt.imshow(speaker_invalid_image)
                # plot_time_series(data, "short")
                #plt.gcf().canvas.mpl_connect('key_press_event', close_figure)
                #plt.show()

        else:
            print("000000000")  # 기동어 들을 준비 완료


    else:
        frame.append(temp_data)
        print(np.array(frame).shape)


stream_wuw.stop_stream()
stream_wuw.close()
p.terminate()
"""