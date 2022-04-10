import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

import tensorflow as tf
from tensorflow import lite
import tempfile
from configuration import Config

# tflite 모델 평가에서 사용되는 실행 함수
def run_tflite_model(tflite_file, test_image_indices, x_test, y_test):
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

        predictions[i] = output.argmax()

    return predictions, fps

def evaluate_tflite_model(target,t_list ,tflite_file, model_type,  x_test, y_test, model_name, x_train, y_train, x_val, y_val):

  test_image_indices = range(x_test.shape[0])

  predictions, fps = run_tflite_model(tflite_file, test_image_indices, x_test, y_test)
  accuracy = (np.sum(y_test.reshape(-1)== predictions) * 100) / len(x_test)
  confusion_mtx = tf.math.confusion_matrix(y_test, predictions)
  plt.figure(figsize=(5, 5))
  sns.heatmap(confusion_mtx,
            annot=True,
            xticklabels = t_list,
            yticklabels = t_list,
            cmap="Reds")
  plt.savefig(Config.base_path + 'model_evaluate/' + model_name)
  plt.show()


  target_count = 0
  def count_target(target, t_list, y_data):
      count_result = []
      for t in target:
          count = 0

          for c in y_data:
              if t_list[int(c)] == t:
                  count += 1
          count_result.append(count)
      return count_result
  result = count_target(target, t_list, y_test)

  print("-"*50)
  print(model_name)
  print("target : ", target)
  print("target의 개수 : ",result )
  print("데이터 개수 : ",len(y_test))
  print("FPS : ", 1 / np.mean(fps))
  # print("Thres - hold : {}".format(Config.thres_hold))
  print('Test Accureacy: {:0.02f}% '.format(metrics.accuracy_score( y_test,predictions)*100))
  # print('Test Precision: ',metrics.precision_score( y_test,predictions))
  # print('Test Recall: ',metrics.recall_score( y_test,predictions ))
  # print('Test F1 score: ',metrics.f1_score(y_test,predictions ))

def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile
  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)
  print("File_size : {:.2f} KB".format(float(os.path.getsize(zipped_file))/1024))



feature_sets_detect = np.load( Config.base_path + "spec_set_multi.npz")
x_train_detect = feature_sets_detect['x_train']
y_train_detect = feature_sets_detect['y_train']
x_val_detect = feature_sets_detect['x_val']
y_val_detect = feature_sets_detect['y_val']
x_test_detect = feature_sets_detect['x_test']
y_test_detect = feature_sets_detect['y_test']


feature_sets_recog = np.load( Config.base_path + "user_set_multi.npz")
x_train_recog = feature_sets_recog['x_train']
y_train_recog = feature_sets_recog['y_train']
x_val_recog = feature_sets_recog['x_val']
y_val_recog = feature_sets_recog['y_val']
x_test_recog = feature_sets_recog['x_test']
y_test_recog = feature_sets_recog['y_test']

print("detect 관련")
detect_model = tf.keras.models.load_model(Config.best_model_path)
_, detect_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(detect_model, detect_keras_file, include_optimizer=False)
get_gzipped_model_size(detect_keras_file)
get_gzipped_model_size(Config.tflite_file_path)
get_gzipped_model_size(Config.prun_02_tflite_file_path)
get_gzipped_model_size(Config.prun_04_tflite_file_path)
get_gzipped_model_size(Config.prun_06_tflite_file_path)
get_gzipped_model_size(Config.prun_08_tflite_file_path)


model_name_detect = "tflite_orig_detect_wuw_matrix.jpg"
model_type = "Float"
evaluate_tflite_model(Config.target_wake_word, Config.target_list,
                      Config.tflite_file_path, model_type,x_test_detect,
                      y_test_detect, model_name_detect,
                      x_train_detect, y_train_detect, x_val_detect, y_val_detect
                      )

model_name_detect = "tflite_pruning_02_detect_wuw_matrix.jpg"
model_type = "Float"
evaluate_tflite_model(Config.target_wake_word, Config.target_list,
                      Config.prun_02_tflite_file_path, model_type,x_test_detect,
                      y_test_detect, model_name_detect,
                      x_train_detect, y_train_detect, x_val_detect, y_val_detect
                      )

model_name_detect = "tflite_pruning_04_detect_wuw_matrix.jpg"
model_type = "Float"
evaluate_tflite_model(Config.target_wake_word, Config.target_list,
                      Config.prun_04_tflite_file_path, model_type,x_test_detect,
                      y_test_detect, model_name_detect,
                      x_train_detect, y_train_detect, x_val_detect, y_val_detect
                      )

model_name_detect = "tflite_pruning_06_detect_wuw_matrix.jpg"
model_type = "Float"
evaluate_tflite_model(Config.target_wake_word, Config.target_list,
                      Config.prun_06_tflite_file_path, model_type,x_test_detect,
                      y_test_detect, model_name_detect,
                      x_train_detect, y_train_detect, x_val_detect, y_val_detect
                      )

model_name_detect = "tflite_pruning_08_detect_wuw_matrix.jpg"
model_type = "Float"
evaluate_tflite_model(Config.target_wake_word, Config.target_list,
                      Config.prun_08_tflite_file_path, model_type,x_test_detect,
                      y_test_detect, model_name_detect,
                      x_train_detect, y_train_detect, x_val_detect, y_val_detect
                      )


print()
print("-"*100)
print()

print("Recognition 관련")
recog_model = tf.keras.models.load_model(Config.best_model_path_recog_02)
_, recog_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(recog_model, recog_keras_file, include_optimizer=False)

get_gzipped_model_size(recog_keras_file)
get_gzipped_model_size(Config.tflite_file_path_recog_02)

get_gzipped_model_size(Config.prun_02_tflite_file_path_recog)
get_gzipped_model_size(Config.prun_04_tflite_file_path_recog)
get_gzipped_model_size(Config.prun_06_tflite_file_path_recog)
get_gzipped_model_size(Config.prun_08_tflite_file_path_recog)


model_name_recog = "tflite_orig_recog_user_matrix_02.jpg"
model_type = "Float"
evaluate_tflite_model(Config.target_user, Config.user_list,
                      Config.tflite_file_path_recog_02, model_type, x_test_recog,
                      y_test_recog, model_name_recog,
                      x_train_recog, y_train_recog, x_val_recog, y_val_recog
                      )


model_name_recog = "tflite_pruning_02_recog_user_matrix.jpg"
model_type = "Float"
evaluate_tflite_model(Config.target_user, Config.user_list,
                      Config.prun_02_tflite_file_path_recog, model_type, x_test_recog,
                      y_test_recog, model_name_recog,
                      x_train_recog, y_train_recog, x_val_recog, y_val_recog
                      )

model_name_recog = "tflite_pruning_04_recog_user_matrix.jpg"
model_type = "Float"
evaluate_tflite_model(Config.target_user, Config.user_list,
                      Config.prun_04_tflite_file_path_recog, model_type, x_test_recog,
                      y_test_recog, model_name_recog,
                      x_train_recog, y_train_recog, x_val_recog, y_val_recog
                      )

model_name_recog = "tflite_pruning_06_recog_user_matrix.jpg"
model_type = "Float"
evaluate_tflite_model(Config.target_user, Config.user_list,
                      Config.prun_06_tflite_file_path_recog, model_type, x_test_recog,
                      y_test_recog, model_name_recog,
                      x_train_recog, y_train_recog, x_val_recog, y_val_recog
                      )

model_name_recog = "tflite_pruning_08_recog_user_matrix.jpg"
model_type = "Float"
evaluate_tflite_model(Config.target_user, Config.user_list,
                      Config.prun_08_tflite_file_path_recog, model_type, x_test_recog,
                      y_test_recog, model_name_recog,
                      x_train_recog, y_train_recog, x_val_recog, y_val_recog
                      )






"""
evaluate_tflite_model(Config.quant_tflite_file_path, model_type="Quantization")
get_gzipped_model_size(Config.quant_tflite_file_path)

evaluate_tflite_model(Config.prun_tflite_file_path, model_type="Pruning")
get_gzipped_model_size(Config.prun_tflite_file_path)
"""

# 세로 - True
# 가로 - Prediction