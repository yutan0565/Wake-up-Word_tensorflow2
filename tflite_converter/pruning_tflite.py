
from tensorflow.keras.models import load_model
import tensorflow_model_optimization as tfmot
from tensorflow import lite
import tensorflow as tf
from configuration import Config
import numpy as np

import tempfile

feature_sets = np.load( Config.base_path + "mfcc_set.npz")

# 저장되어 있는 mfcc feature 들 불러 오기
x_train = feature_sets['x_train']
y_train = feature_sets['y_train']
x_val = feature_sets['x_val']
y_val = feature_sets['y_val']
x_test = feature_sets['x_test']
y_test = feature_sets['y_test']

model = tf.keras.models.load_model(Config.best_model_path)
_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model, keras_file, include_optimizer=False)

# Pruning을 위한 변수 설정
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
num_images = x_train.shape[0] * (1 - Config.validation_split_prun)
end_step = np.ceil(num_images / Config.batch_size_prun).astype(np.int32) * Config.epoch_prun

# Sparsity = 0.6,   pruning하고 다시 학습 시키기
pruning_params_6 = {
      'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=0.6,
                                                               begin_step=0,
                                                               end_step=-1,
                                                               frequency = 10
                                                               )
}

wake_pruning_6 = prune_low_magnitude(model, **pruning_params_6)

wake_pruning_6.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

logdir = tempfile.mkdtemp()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

wake_pruning_6.fit(x_train, y_train,
                  batch_size=Config.batch_size_prun, epochs=Config.epoch_prun, validation_split=Config.validation_split_prun,
                  callbacks=callbacks)

pruning_6_eport = tfmot.sparsity.keras.strip_pruning(wake_pruning_6)

converter = tf.lite.TFLiteConverter.from_keras_model(pruning_6_eport)
pruning_6_tflite = converter.convert()

open(Config.prun_tflite_file_path, "wb") .write(pruning_6_tflite)