from tensorflow.keras.models import load_model
import tensorflow_model_optimization as tfmot
from tensorflow import lite
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard

import tensorflow as tf
from configuration import Config
import numpy as np

import tempfile


feature_sets = np.load( Config.base_path + "spec_set_multi.npz")
#feature_sets = np.load( Config.base_path + "user_set_multi.npz")

# 저장되어 있는 mfcc feature 들 불러 오기
x_train = feature_sets['x_train']
y_train = feature_sets['y_train']
x_val = feature_sets['x_val']
y_val = feature_sets['y_val']
x_test = feature_sets['x_test']
y_test = feature_sets['y_test']

sample_shape = x_test.shape[1:]
print(sample_shape)

y_train_one_hot = tf.one_hot(y_train, len(Config.target_list))
y_val_one_hot = tf.one_hot(y_val, len(Config.target_list))
y_test_one_hot = tf.one_hot(y_test, len(Config.target_list))

model = tf.keras.models.load_model(Config.best_model_path)

best_model_path_list = [Config.best_model_path_detect_pruning_02, Config.best_model_path_detect_pruning_04, Config.best_model_path_detect_pruning_06, Config.best_model_path_detect_pruning_08 ]
tflite_path_list = [Config.prun_02_tflite_file_path, Config.prun_04_tflite_file_path, Config.prun_06_tflite_file_path, Config.prun_08_tflite_file_path]
initial_list =  [ 0.00, 0.20, 0.40, 0.60 ]
final_list = [ 0.20, 0.40, 0.60, 0.80 ]

for i in range(4):
    best_model_path = best_model_path_list[i]
    tflite_path = tflite_path_list[i]
    initial = initial_list[i]
    final = final_list[i]

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    end_step = np.ceil(x_train.shape[0]  / Config.batch_size_prun).astype(np.int32) * Config.epoch_prun

    pruning_params= {
          'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=initial,
                                                                   final_sparsity=final,
                                                                   begin_step=0,
                                                                   end_step=end_step
                                                                   )
    }

    def check_layers_name(model):
        for n in range(len(model.layers)):
            print(model.layers[n].name)

    # 첫번째랑, 마지막 부분은 제외하고 Pruning 진행 할거임
    # def custum_pruning_apply(layer):
    #     if tf.keras.layer.name != "conv2d" or tf.keras.layer.name != "dense_3":
    #         return prune_low_magnitude(layer, **pruning_params)
    #     return layer

    def custum_pruning_apply(layer):
        if layer.name != "conv2d" or layer.name != "dense_3":
            return prune_low_magnitude(layer, **pruning_params)
        return layer

    # 원래 모델
    model.summary()
    check_layers_name(model)

    model_for_pruning = tf.keras.models.clone_model(
        model,
        clone_function = custum_pruning_apply
    )
    model_for_pruning.summary()



    opt = tf.keras.optimizers.Adam(learning_rate=Config.start_lr)
    model_for_pruning.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    # logdir = tempfile.mkdtemp()




    early_stop = EarlyStopping(patience=Config.early_stop_aptience)
    mc = ModelCheckpoint(best_model_path,
                         save_best_only=True,
                         monitor = 'val_loss',
                         verbose = 1,
                         mode = 'min')

    reduce_lr  = ReduceLROnPlateau(monitor = 'val_loss',
                                   factor=Config.lr_factor,
                                   patience=Config.lr_patience
                                   )

    callbacks = [
      tfmot.sparsity.keras.UpdatePruningStep(),
      # tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
      early_stop,
      mc,
      reduce_lr
    ]

    model_for_pruning.fit(x_train, y_train_one_hot,
                      batch_size=Config.batch_size_prun,
                      epochs=Config.epoch_prun,
                      verbose = 1,
                      validation_data = (x_val, y_val_one_hot),
                      callbacks=callbacks)




    baseline_model_accuracy = model.evaluate(x_test, y_test_one_hot, verbose=0)

    _, model_for_pruning_accuracy = model_for_pruning.evaluate(x_test, y_test_one_hot, verbose=0)

    print('Baseline test accuracy:', baseline_model_accuracy)
    print('Pruned test accuracy:', model_for_pruning_accuracy)


    pruning_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    converter = tf.lite.TFLiteConverter.from_keras_model(pruning_export)
    pruning_tflite = converter.convert()


    open(tflite_path, "wb") .write(pruning_tflite)