from os import listdir
import random
import numpy as np
import librosa
import audio_tool as tool

from configuration import Config
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


## 데이터 분리
filenames = []
y = []

for user in Config.user_list:
    for index, target in enumerate(Config.target_list):
        # 폴더 추가 해야함
        filenames.append(listdir('/'.join([Config.aug_dataset_path, user, target])))
        y.append(np.ones(len(filenames[index])) * index)

# 하나로 쭉 나열 하기
filenames = [item for sublist in filenames for item in sublist]
y = [item for sublist in y for item in sublist]


# 여기부터 분배를 잘 해줘야함
# file 모아둔거 한번 섞어 주기
filenames_y = list(zip(filenames, y))
random.shuffle(filenames_y)
filenames, y = zip(*filenames_y)



val_set_size = int(len(filenames) * Config.val_ratio)
test_set_size = int(len(filenames) * Config.test_ratio)

filenames_val = filenames[:val_set_size]
filenames_test = filenames[val_set_size:(val_set_size + test_set_size)]
filenames_train = filenames[(val_set_size + test_set_size):]

y_orig_val = y[:val_set_size]
y_orig_test = y[val_set_size:(val_set_size + test_set_size)]
y_orig_train = y[(val_set_size + test_set_size):]

print(len(filenames_train), len(y_orig_train))
print(len(filenames_val), len(y_orig_val))
print(len(filenames_test), len(y_orig_test))

for i in range(100):
    print(filenames_train[i],y_orig_train[i] )


def extract_features(in_files, in_y):
    prob_cnt = 0
    out_x = []
    out_y = []
    for index, filename in enumerate(in_files):
        for user in Config.user_list:
            # Create path from given filename and target item
            #
            if (user not in filename) or (Config.target_list[int(in_y[index])] not in filename):
                continue
            if Config.target_list[int(in_y[index])] != 'other_google_speech':
                if (Config.target_list[int(in_y[index])] != 'other') and ('other' in filename ):
                    continue



            path = "/".join([Config.aug_dataset_path, user,Config.target_list[int(in_y[index])],
                             filename])

            # Check to make sure we're reading a .wav file
            if not path.endswith('.wav'):
                continue
            print(filename, int(in_y[index]))
            # Create MFCCs
            signal, sr = librosa.core.load(path, Config.sample_rate)
            if len(signal) < 32000:
                print('Dropped:', index, signal.shape)
                prob_cnt += 1
                continue
            #signal = signal[int(-Config.sample_cut - Config.click): int(-Config.click)]
            #mfccs = MFCC_maker.mfcc_process (signal, sr)
            spectrogram = tool.mel_spectrogram_process(signal, sr)
            regul_spectrogram = tool.spec_regularization(spectrogram)
            if regul_spectrogram.shape[1] == Config.len_mfcc:
                out_x.append(regul_spectrogram)
                out_y.append(in_y[index])
            else:
                print('Dropped:', index, regul_spectrogram.shape)
                prob_cnt += 1

    return out_x, out_y, prob_cnt

x_train, y_train, prob_train = extract_features(filenames_train, y_orig_train)
x_val, y_val, prob_val = extract_features(filenames_val, y_orig_val)
x_test, y_test, prob_test = extract_features(filenames_test, y_orig_test)

print("Train 잃은거{}".format(prob_train / len(y_orig_train)))
print("Valid 잃은거{}".format(prob_val / len(y_orig_val)))
print("Test 잃은거{}".format(prob_test / len(y_orig_test)))

#wake_word_index = Config.target_list.index(Config.wake_word)

# y_train = np.equal(y_train, wake_word_index).astype('float64')
# y_val = np.equal(y_val, wake_word_index).astype('float64')
# y_test = np.equal(y_test, wake_word_index).astype('float64')


# CNN 에 넣기 이전에 Channel을 1로 만들어주기

x_train =  np.array(x_train)
x_val =  np.array(x_val)
x_test =  np.array(x_test)


for i in range(100):
    print(x_train[i],y_train[i] )

x_train = x_train.reshape(x_train.shape[0],
                          x_train.shape[1],
                          x_train.shape[2],
                          1)
x_val =  x_val.reshape(x_val.shape[0],
                      x_val.shape[1],
                      x_val.shape[2],
                      1)
x_test =  x_test.reshape(x_test.shape[0],
                        x_test.shape[1],
                        x_test.shape[2],
                        1)
print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

np.savez(Config.base_path +"spec_set_multi.npz",
         x_train=x_train,
         y_train=y_train,
         x_val=x_val,
         y_val=y_val,
         x_test=x_test,
         y_test=y_test)

# # input shape 설정 해주기
# sample_shape = x_test.shape[1:]
# print(sample_shape)


# def plot_time_series(data, title):
#     fig = plt.figure(figsize=(7, 4))
#     plt.title(title+'  wave')
#     plt.ylabel('Amplitude')
#     plt.plot(np.linspace(0, 5, len(data)), data)
#     plt.show()
#
# def show_aug_sound(file_path, type):
#     sig, sr = librosa.load(file_path, sr=16000)
#     print('sr:', sr, ', sig shape:', sig.shape)
#     print('length:', sig.shape[0] / float(sr), 'secs')
#
#     plot_time_series(sig, "original")