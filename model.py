from keras.layers import Dense,  Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras import Sequential
from configuration import Config

def cnn_wuw_detection_binary_model_01(sample_shape):
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
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation="sigmoid")
    ])
    model = Sequential([conv_layer,
                        fc_layer
                        ])
    return model

def cnn_wuw_detection_binary_model_02(sample_shape):
    conv_layer = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=sample_shape, padding = 'valid'),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(128, (3, 3), activation='relu', padding = 'valid'),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(256, (3, 3), activation='relu', padding = 'valid'),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(512, (3, 3), activation='relu', padding = 'valid'),
        MaxPooling2D(pool_size=(3, 3))
    ])
    fc_layer = Sequential([
        Flatten(),
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        Dense(64, activation='relu'),
        # Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])


    model = Sequential([conv_layer,
                        fc_layer
                        ])
    return model
def cnn_wuw_detection_multi_model(sample_shape):
    conv_layer = Sequential([
        Conv2D(32, (2, 2), activation='relu', input_shape=sample_shape ),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (2, 2), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (2, 2), activation='relu'),
        MaxPooling2D(pool_size=(2, 2))
    ])
    fc_layer = Sequential([
        Flatten(),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(len(Config.target_list), activation="softmax")
    ])
    model = Sequential([conv_layer,
                        fc_layer
                        ])
    return model

def cnn_wuw_detection_multi_model_02(sample_shape):
    conv_layer = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=sample_shape ),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(64, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(3, 3))
    ])
    fc_layer = Sequential([
        Flatten(),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(len(Config.target_list), activation="softmax")
    ])
    model = Sequential([conv_layer,
                        fc_layer
                        ])
    return model

