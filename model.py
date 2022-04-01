from tensorflow.keras import layers
from keras.layers import Dense,  Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Model

class MyModel(Model):
    def __init__(self, sample_shape):
        super(MyModel, self).__init__()

        self.sample_shape = sample_shape

        self.conv1 = Conv2D(32, (2, 2), activation='relu', input_shape= self.sample_shape)
        self.maxpool1 = MaxPooling2D(pool_size=(2, 2))
        self.conv2 = Conv2D(64, (2, 2), activation='relu')
        self.maxpool2 = MaxPooling2D(pool_size=(2, 2))
        self.conv3 = Conv2D(128, (2, 2), activation='relu')
        self.maxpool3 = MaxPooling2D(pool_size=(2, 2))
        self.flatten = Flatten()

        self.dense1 = Dense(256, activation='relu')
        self.dense2 = Dense(256, activation='relu')
        self.drop = Dropout(0.5)
        self.dense3 = Dense(64, activation='relu')
        self.dense4 = Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.drop(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x

    def summary(self):
        inputs = Input( self.sample_shape)
        Model(inputs, self.call(inputs)).summary()

model = MyModel((20, 38, 1))
model.summary()