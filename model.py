from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, LeakyReLU
from tensorflow.keras import Sequential
from configuration import Config

#######
#Trainable params: 372,166
def detection_model_01(sample_shape):
    model = Sequential([
        # conv layer 부분
        Conv2D(32, (3, 3), activation='relu', input_shape=sample_shape),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(64, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(3, 3)),

        # FC layer 부분
        Flatten(),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(len(Config.target_list), activation="softmax")
    ])

    return model


# 모델 크기가 작은 거 !!!
def detection_model_02(sample_shape):
    model = Sequential([
        # conv layer 부분
        Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.3), input_shape=sample_shape),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.3)),
        MaxPooling2D(pool_size=(3, 3)),

        # FC layer 부분
        Flatten(),
        Dense(64, activation=LeakyReLU(alpha=0.3)),
        Dropout(0.5),
        Dense(64, activation=LeakyReLU(alpha=0.3)),
        Dense(len(Config.target_list), activation="softmax")
    ])
    return model


# 인식 전용 모델
#######
# Trainable params: 565,731
def recog_model_01(sample_shape):
    model = Sequential([
        # conv layer 부분
        Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.3), input_shape=sample_shape, ),
        Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.3)),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.3), padding='same'),
        Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.3)),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.3), padding='same'),
        Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.3)),
        MaxPooling2D(pool_size=(3, 3)),

        # FC layer 부분
        Flatten(),
        Dense(256, activation=LeakyReLU(alpha=0.3)),
        Dense(256, activation=LeakyReLU(alpha=0.3)),
        Dropout(0.5),
        Dense(64, activation=LeakyReLU(alpha=0.3)),
        Dense(len(Config.user_list), activation="softmax")
    ])
    return model

# Trainable params: 730,563
def recog_model_02(sample_shape):
    model = Sequential([
        # conv layer 부분
        Conv2D(32, (4, 2), activation=LeakyReLU(alpha=0.3), input_shape=sample_shape, ),
        Conv2D(32, (4, 2), activation=LeakyReLU(alpha=0.3)),
        MaxPooling2D(pool_size=(4, 2)),
        Conv2D(64, (4, 2), activation=LeakyReLU(alpha=0.3), padding='same'),
        Conv2D(64, (4, 2), activation=LeakyReLU(alpha=0.3)),
        MaxPooling2D(pool_size=(4, 2)),
        Conv2D(128, (4, 2), activation=LeakyReLU(alpha=0.3), padding='same'),
        Conv2D(128, (4, 2), activation=LeakyReLU(alpha=0.3)),
        MaxPooling2D(pool_size=(4, 2)),

        # FC layer 부분
        Flatten(),
        Dense(256, activation=LeakyReLU(alpha=0.3)),
        Dense(256, activation=LeakyReLU(alpha=0.3)),
        Dropout(0.5),
        Dense(64, activation=LeakyReLU(alpha=0.3)),
        Dense(len(Config.user_list), activation="softmax")
    ])
    return model