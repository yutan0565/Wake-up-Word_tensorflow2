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


#model = tf.keras.models.load_model(Config.best_model_path_recog)

_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model, keras_file, include_optimizer=False)

# Pruning을 위한 변수 설정
# wrap  씌어 줄 준비
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
num_images = x_train.shape[0] * (1 - Config.validation_split_prun)
end_step = np.ceil(num_images / Config.batch_size_prun).astype(np.int32) * Config.epoch_prun

# Sparsity = 0.6,   pruning하고 다시 학습 시키기
pruning_params= {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step
                                                               )
}

detection_pruning= prune_low_magnitude(model, **pruning_params)

opt = tf.keras.optimizers.Adam(learning_rate=Config.start_lr)
detection_pruning.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
logdir = tempfile.mkdtemp()


early_stop = EarlyStopping(patience=Config.early_stop_aptience)
mc = ModelCheckpoint(Config.best_model_path_detect_pruning,
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
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
  early_stop,
  mc,
  reduce_lr
]

detection_pruning.fit(x_train, y_train,
                  batch_size=Config.batch_size_prun, epochs=Config.epoch_prun,
                  validation_split=Config.validation_split_prun, callbacks=callbacks)

pruning_export = tfmot.sparsity.keras.strip_pruning(detection_pruning)

converter = tf.lite.TFLiteConverter.from_keras_model(pruning_export)
pruning_tflite = converter.convert()

open(Config.prun_tflite_file_path, "wb") .write(pruning_tflite)