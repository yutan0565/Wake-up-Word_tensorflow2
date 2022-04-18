import pyaudio
import playsound
import soundfile as sf
import librosa.display

import audio_tool
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

# for index in range(audio.get_device_count()):
#     desc = audio.get_device_info_by_index(index)
#     print("DEVICE: {device}, INDEX: {index}, RATE: {rate} ".format(
#         device=desc["name"], index=index, rate=int(desc["defaultSampleRate"])))

CHUNK = int(Config.sample_cut / Config.stride_rate) # 8000

#tflite_model_path_detect = Config.prun_08_tflite_file_path  ###
tflite_model_path_detect = Config.tflite_file_path  ###
interpreter_detect = tflite.Interpreter(tflite_model_path_detect)
interpreter_detect.allocate_tensors()
input_details_detect = interpreter_detect.get_input_details()
output_details_detect = interpreter_detect.get_output_details()

detection_image_path = Config.base_path +"show_image/Detection.png"
detection_image_pil = Image.open(detection_image_path)
detection_image = np.array(detection_image_pil)

speaker_valid_sound_path = Config.base_path + 'show_image/검증_성공.mp3'
speaker_invalid_sound_path = Config.base_path + 'show_image/검증_실패.mp3'

audio = pyaudio.PyAudio()
p = pyaudio.PyAudio()

stream_wuw = p.open(format=pyaudio.paFloat32, channels=1, rate=Config.sample_rate, input=True,
                frames_per_buffer=CHUNK, input_device_index=1)

frame = []
initiate_frame = [0 for i in range(CHUNK)]

frame.append(initiate_frame)
low_power_flag = True
thres_hold_low_power = 0.002

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
    plt.imshow(result_image)
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


print("프로그램 시작!!")


listen_count = 0

while (True):
    temp_data = np.fromstring(stream_wuw.read(CHUNK), dtype=np.float32)
    # print("{:.05f}".format(np.mean(np.abs(temp_data))))
    listen_count -= 1

    if np.mean(np.abs(temp_data)) <  0.01: #thres_hold_low_power:
        if np.array(frame).shape[0] == Config.stride_rate:
            frame, _ = make_frame(frame, temp_data)
        else:
            frame.append(temp_data)
        if low_power_flag == True and listen_count == 0:
            #print("저전력 모드!!")  # 기동어 인식 안하고 있는 상태
            print(".")
            low_power_flag = False
        continue

    if np.array(frame).shape[0] == Config.stride_rate:
        frame, signal = make_frame(frame, temp_data)

        signal = audio_tool.max_scaler(signal)
        spectrogram = tool.mel_spectrogram_process(signal, Config.sample_rate)
        regul_spectrogram = tool.spec_regularization(spectrogram)
        input_tensor = regul_spectrogram.reshape(1, regul_spectrogram.shape[0], regul_spectrogram.shape[1] ,1)

        val_detect = make_ouput( input_tensor ,interpreter_detect,input_details_detect, output_details_detect )
        val_detect_true = val_detect[4]
        val_detect_false = val_detect[0] + val_detect[1] + val_detect[2] + val_detect[3] + val_detect[5] + val_detect[6]

        if val_detect_true > Config.thres_hold_detect:

            # print("Hi Yutan 확률 : {:.02f}".format(val_detect_true))
            print("{} 감지!!".format(Config.target_list[4]))

            # plt.figure(figsize=(12, 4))
            # librosa.display.specshow(regul_spectrogram, sr=Config.sample_rate, x_axis='time', y_axis='mel')
            # plt.title('regul_spectrogram')
            # plt.colorbar(format='%+02.0f dB')
            # plt.tight_layout()

            show_result_image(detection_image)
            frame = []

        else:
            listen_count = 15
            if low_power_flag == False:
                print("------------듣는중----------- ")  # 기동어 들을 준비 완료
                low_power_flag = True

    else:
        frame.append(temp_data)



stream_wuw.stop_stream()
stream_wuw.close()
p.terminate()
