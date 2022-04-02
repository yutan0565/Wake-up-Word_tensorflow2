
from keras.layers import Dense,  Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten

from keras import Sequential


def cnn_wuw_detection_model(sample_shape):
    conv_layer = Sequential([
        Conv2D(32, (2, 2), activation='relu', input_shape=sample_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (2, 2), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (2, 2), activation='relu'),
        MaxPooling2D(pool_size=(2, 2))
    ])
    fc_layer = Sequential([
        Flatten(),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])


    model = Sequential([conv_layer,
                        fc_layer
                        ])
    return model


model = cnn_wuw_detection_model((20, 38, 1))
model.summary()