import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn import metrics
import tensorflow as tf

from configuration import Config

feature_sets_detect = np.load( Config.base_path + "spec_set_multi.npz")
x_test_detect = feature_sets_detect['x_test']
y_test_detect = feature_sets_detect['y_test']
print(y_test_detect)

feature_sets_recog = np.load( Config.base_path + "user_set_multi.npz")
x_test_recog = feature_sets_recog['x_test']
y_test_recog = feature_sets_recog['y_test']
print(y_test_recog)

def evaluate(model, x_test, y_test, model_name, label):
    Y_pred = model.predict(x_test)
    predictions = []

    for result in Y_pred:
        a = np.array(result)
        predictions.append(a.argmax())
    confusion_mtx = metrics.confusion_matrix(y_test, predictions)

    sns.heatmap(confusion_mtx,
                annot=True,
                xticklabels=label,  # label,
                yticklabels=label,  # label,
                cmap='Blues')
    plt.savefig(Config.base_path + 'model_evaluate/' + model_name )
    plt.show()
    print("Thres - hold : {}".format(Config.thres_hold))
    print('Test Accureacy: ', metrics.accuracy_score(y_test, predictions))
    # print('Test Precision: ', metrics.precision_score(y_test, predictions))
    # print('Test Recall: ', metrics.recall_score(y_test, predictions))
    # print('Test F1 score: ', metrics.f1_score(y_test, predictions))
    # print("Test 에서 target 의 비율 : {}%".format(round(list(y_test).count(1) / len(y_test) * 100, 2)))

model_name_detect = "detect_wuw_matrix.jpg"
model_detect = tf.keras.models.load_model(Config.best_model_path)
evaluate(model_detect,x_test_detect, y_test_detect, model_name_detect, Config.target_list)

model_name_recog = "recog_user_matrix_02.jpg"
model_recog = tf.keras.models.load_model(Config.best_model_path_recog_02)
evaluate(model_recog, x_test_recog, y_test_recog, model_name_recog, Config.user_list)
