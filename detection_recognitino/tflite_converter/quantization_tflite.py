from tensorflow.keras.models import load_model
import tensorflow_model_optimization as tfmot
from tensorflow import lite
import tensorflow as tf
from configuration import Config
import numpy as np

feature_sets = np.load( Config.base_path + "mfcc_set.npz")

# 저장되어 있는 mfcc feature 들 불러 오기
x_train = feature_sets['x_train']

# Quantization을 위한 point 설정
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(300):
    yield [input_value]

model = tf.keras.models.load_model(Config.best_model_path)

# int 8 로 quantization 진행하기
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model_quant = converter.convert()

open(Config.quant_tflite_file_path, 'wb').write(tflite_model_quant)

interpreter = tf.lite.Interpreter(model_path = Config.quant_tflite_file_path)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)

