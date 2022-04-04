import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

import tensorflow as tf
from tensorflow import lite
import tempfile
from configuration import Config

np.random.seed(77)

feature_sets = np.load( Config.base_path + "mfcc_set_multi.npz")

# 저장되어 있는 mfcc feature 들 불러 오기

x_train = feature_sets['x_train']
y_train = feature_sets['y_train']
x_val = feature_sets['x_val']
y_val = feature_sets['y_val']
x_test = feature_sets['x_test']
y_test = feature_sets['y_test']

# tflite 모델 평가에서 사용되는 실행 함수
def run_tflite_model(tflite_file, test_image_indices):
    global x_test
    global y_test

    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    fps = np.zeros((len(test_image_indices),))
    predictions = np.zeros((len(test_image_indices),), dtype=int)

    for i, test_image_index in enumerate(test_image_indices):
        test_image = x_test[test_image_index]
        test_label = y_test[test_image_index]

        # Check if the input type is quantized, then rescale input data to uint8
        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_image / input_scale + input_zero_point

        start = time.time()
        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        end = time.time()

        fps[i] = (end - start)
        # thres_hold 설정 해주기

        if output > Config.thres_hold:
            output = 1
        else:
            outut = 0

        predictions[i] = output

    return predictions, fps

def evaluate_tflite_model(tflite_file, model_type):
  global x_test
  global x_test

  test_image_indices = range(x_test.shape[0])
  predictions, fps = run_tflite_model(tflite_file, test_image_indices)
  accuracy = (np.sum(y_test.reshape(-1)== predictions) * 100) / len(x_test)

  confusion_mtx = tf.math.confusion_matrix(y_test, predictions)

  print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (model_type, accuracy, len(x_test)))

  sns.heatmap(confusion_mtx,
            annot=True,
            xticklabels = Config.label,
            yticklabels = Config.label,
            cmap='Blues')
  plt.show()

  print("FPS : ", 1 / np.mean(fps))
  print("Thres - hold : {}".format(Config.thres_hold))
  print('Test Accureacy: ',metrics.accuracy_score( y_test,predictions))
  print('Test Precision: ',metrics.precision_score( y_test,predictions))
  print('Test Recall: ',metrics.recall_score( y_test,predictions ))
  print('Test F1 score: ',metrics.f1_score(y_test,predictions ))
  print("Train 에서 1의 비율 : {}%".format(  round(list(y_train).count(1) / len(y_train)*100 , 2 )))
  print("Val 에서 1의 비율 : {}%".format(  round(list(y_val).count(1) / len(y_val)*100 , 2 )))
  print("Test 에서 1의 비율 : {}%".format(  round(list(y_test).count(1) / len(y_test)*100 , 2 )))

def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile
  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)
  print("File_size : {:.2f} KB".format(float(os.path.getsize(zipped_file))/1024))


evaluate_tflite_model(Config.tflite_file_path, model_type="Float")
get_gzipped_model_size(Config.tflite_file_path)

"""
evaluate_tflite_model(Config.quant_tflite_file_path, model_type="Quantization")
get_gzipped_model_size(Config.quant_tflite_file_path)

evaluate_tflite_model(Config.prun_tflite_file_path, model_type="Pruning")
get_gzipped_model_size(Config.prun_tflite_file_path)
"""

# 세로 - True
# 가로 - Prediction