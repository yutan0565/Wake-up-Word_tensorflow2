import librosa
import numpy as np

sig, sr = librosa.load("C:/Users/yutan/Desktop/Wake-up-Word_tensorflow2/custum_dataset/user_02/hi_byeonghyeon/ori_user_02_0002_hi_byeonghyeon.wav", sr=16000)
print(min(sig))
print(max(sig))
