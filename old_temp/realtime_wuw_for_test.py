import pyaudio
import playsound

from configuration import Config
import audio_tool as tool
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

audio = pyaudio.PyAudio()

for index in range(audio.get_device_count()):
    desc = audio.get_device_info_by_index(index)
    print("DEVICE: {device}, INDEX: {index}, RATE: {rate} ".format(
        device=desc["name"], index=index, rate=int(desc["defaultSampleRate"])))


CHUNK = int(Config.sample_cut / Config.stride_rate) # 8000


tflite_model_path_detect = Config.tflite_file_path
#tflite_model_path_detect = Config.prun_99_tflite_file_path  ###

interpreter_detect = tflite.Interpreter(tflite_model_path_detect)
interpreter_detect.allocate_tensors()
input_details_detect = interpreter_detect.get_input_details()
output_details_detect = interpreter_detect.get_output_details()


#tflite_model_path_recog = Config.tflite_file_path_recog_02 ###
tflite_model_path_recog = Config.prun_99_tflite_file_path_recog ###

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

speaker_valid_sound_path = Config.base_path + 'show_image/검증_성공.mp3'
speaker_invalid_sound_path = Config.base_path + 'show_image/검증_실패.mp3'


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

detect_count = 0
recog_count = 0

print("프로그램 시작!!")
while (True):
    temp_data = np.fromstring(stream_wuw.read(CHUNK), dtype=np.float32)

    # 기동어 감지 모델을 돌리고 있지 않는 상태
    if np.mean(np.abs(temp_data)) < Config.thres_hold_low_power:
        if np.array(frame).shape[0] == Config.stride_rate:
            frame, _ = make_frame(frame, temp_data)
        else:
            frame.append(temp_data)
        #print("xx")  # 기동어 인식 안하고 있는 상태
        continue

    if np.array(frame).shape[0] == Config.stride_rate:
        frame, signal = make_frame(frame, temp_data)

        spectrogram = tool.mel_spectrogram_process(signal, Config.sample_rate)
        regul_spectrogram = tool.spec_regularization(spectrogram)
        input_tensor = regul_spectrogram.reshape(1, regul_spectrogram.shape[0], regul_spectrogram.shape[1] ,1)

        val_detect = make_ouput( input_tensor ,interpreter_detect,input_details_detect, output_details_detect )

        for index_detect in Config.target_wake_word_index:
            if val_detect[index_detect] > Config.thres_hold_detect:
            #if True:
                print("{} 감지!!".format(Config.target_list[index_detect]))
                detect_count += 1
                val_recog = make_ouput(input_tensor, interpreter_recog, input_details_recog, output_details_recog)
                flag= True
                for index_recog in Config.target_user_index:
                    if val_recog[index_recog] > Config.thres_hold_recog:
                        if flag:
                            print("검증 성공 : 안녕하세요. {} 님!!!".format(Config.user_list[index_recog]))
                            recog_count += 1
                        break
                    else:
                        if flag:
                            print("검증 실패 : 사용자 등록을 해주세요")
                        flag = False
                frame = [] # frame 초기화
                break
            else:
                pass
                #print("000000000")  # 기동어 들을 준비 완료
    else:
        frame.append(temp_data)

    print("검출 수 : {}".format(detect_count))
    print("검증 수 : {}".format(recog_count))

stream_wuw.stop_stream()
stream_wuw.close()
p.terminate()
