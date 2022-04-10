import tensorflow as tf
from configuration import Config

def convert_tflite(model_path, tflite_path ):
    model = tf.keras.models.load_model(model_path)
    model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(tflite_path, 'wb').write(tflite_model)

def show_input_output_type(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)

convert_tflite(Config.best_model_path, Config.tflite_file_path)
convert_tflite(Config.best_model_path_recog_02, Config.tflite_file_path_recog_02)