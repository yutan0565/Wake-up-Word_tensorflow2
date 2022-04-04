
from configuration import Config

import tflite_runtime.interpreter as tflite
import librosa

import MFCC_maker


import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

tflite_model_path = Config.tflite_file_path
interpreter = tflite.Interpreter(tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

data_path = "C:/Users/yutan/Desktop/Wake-up-Word_tensorflow2/custum_dataset/user_01/hi_yutan/ori_user_01_0004_hi_yutan.wav"
signal, sr = librosa.core.load(data_path, Config.sample_rate)
signal = signal[int(-Config.sample_cut - Config.click): int(-Config.click)]
mfccs = MFCC_maker.mel_spectrogram_process(signal,sr )
input_tensor = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)
print(input_tensor.shape)
interpreter.set_tensor(input_details[0]['index'], input_tensor)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
val = output_data[0]
print("{}".format(Config.target_list[val.argmax()]))