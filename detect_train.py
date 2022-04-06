import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn import metrics
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard

import model
from configuration import Config


#feature_sets = np.load( Config.base_path + "mfcc_set_multi.npz")
feature_sets = np.load( Config.base_path + "spec_set_multi.npz")
# 저장되어 있는 mfcc feature 들 불러 오기
x_train = feature_sets['x_train']
y_train = feature_sets['y_train']
x_val = feature_sets['x_val']
y_val = feature_sets['y_val']
x_test = feature_sets['x_test']
y_test = feature_sets['y_test']
print(len(x_train))
print(len(x_val))
print(len(x_test))

sample_shape = x_test.shape[1:]
print(sample_shape)

y_train = tf.one_hot(y_train, len(Config.target_list))
y_val = tf.one_hot(y_val, len(Config.target_list))
y_test = tf.one_hot(y_test, len(Config.target_list))

model = model.detection_model_02(sample_shape)
model.summary()

# Callback 함수 지정 해주기      학습하는 동안 설정해줄것
early_stop = EarlyStopping(patience=Config.early_stop_aptience)
mc = ModelCheckpoint(Config.best_model_path,
                     save_best_only=True,
                     monitor = 'val_loss',
                     verbose = 1,
                     mode = 'min')

reduce_lr  = ReduceLROnPlateau(monitor = 'val_loss',
                               factor=Config.lr_factor,
                               patience=Config.lr_patience
                               )

#optimizer 조정 해주기
opt = tf.keras.optimizers.Adam(learning_rate=Config.start_lr)

# optimizer, loss 함수를 정의하고,  학습 준비를 한다,  metrics 는 어떤 일이 발생하는지 보여줄 것들
#model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])  #categorical_crossentropy   binary_crossentropy
# 한번에 몇개의 데이터 학습하고 가중치 갱신할지
history = model.fit(x_train, y_train,
          epochs= Config.epoch_original,
          verbose=1,
          batch_size=Config.batch_size_original,
          validation_data = (x_val, y_val),
          callbacks = [early_stop, reduce_lr , mc]
          )

fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color = 'b', label = "Training loss")
ax[0].plot(history.history['val_loss'], color='r', label = "validation loss", axes = ax[0])
legend = ax[0].legend(loc='best', shadow = True)

ax[1].plot(history.history['accuracy'], color = 'b', label = "Training accuracy")
ax[1].plot(history.history['val_accuracy'], color = 'r', label = "Validation accuracy")
legend = ax[1].legend(loc = 'best', shadow = True)
plt.show()
# 데이터 불균형으로 인해, val에 있는게 성능이 더 좋아보일 수도 있음


Y_pred = model.predict(x_test)
predictions = []
# for result in Y_pred:
#   if result > Config.thres_hold:
#       predictions.append(1)
#   else:
#       predictions.append(0)
for result in Y_pred:
    a = np.array(result)
    predictions.append(a.argmax())

y_test = feature_sets['y_test']
confusion_mtx = metrics.confusion_matrix(y_test,predictions)

sns.heatmap(confusion_mtx,
            annot=True,
            xticklabels = Config.target_list,#label,
            yticklabels = Config.target_list,#label,
            cmap='Blues')
plt.show()
print("Thres - hold : {}".format(Config.thres_hold))
print('Test Accureacy: ',metrics.accuracy_score( y_test,predictions))
print('Test Precision: ',metrics.precision_score( y_test,predictions))
print('Test Recall: ',metrics.recall_score( y_test,predictions ))
print('Test F1 score: ',metrics.f1_score(y_test,predictions ))
print("Train 에서 1의 비율 : {}%".format(round(list(y_train).count(1) / len(y_train) * 100, 2)))
print("Val 에서 1의 비율 : {}%".format(round(list(y_val).count(1) / len(y_val) * 100, 2)))
print("Test 에서 1의 비율 : {}%".format(round(list(y_test).count(1) / len(y_test) * 100, 2)))
