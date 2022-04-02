from tensorflow.keras.models import load_model
import tensorflow_model_optimization as tfmot
from tensorflow import lite
import tensorflow as tf
from configuration import Config

# 그냥 반
model = tf.keras.models.load_model(Config.best_model_path)
converter = lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(Config.tflite_file_path, 'wb').write(tflite_model)

interpreter = tf.lite.Interpreter(model_path= Config.tflite_file_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)