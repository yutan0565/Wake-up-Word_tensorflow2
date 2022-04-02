from tensorflow.keras import layers
from keras.layers import Dense,  Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Model

class MyModel(Model):
    def __init__(self):  #, sample_shape):
        super(MyModel, self).__init__()

        #self.sample_shape = sample_shape

        self.conv1 = Conv2D(32, (2, 2), activation='relu') #, input_shape= self.sample_shape)
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

    # def summary(self):
    #     inputs = Input( self.sample_shape)
    #     Model(inputs, self.call(inputs)).summary()



loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate = LR)

# loss와 acc 을 누적하고 평균내서 확인 하기
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # Backpropagation을 위한 gradient 값을 gradient에 저장 해두기
    # 모델의 call을 불러 주기
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  # 모든 gradient 값들 정의
  gradients = tape.gradient(loss, model.trainable_variables)
  #Backpropagation 진행 하기, 각각에 상을 하는거 수행
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def val_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  v_loss = loss_object(labels, predictions)

  val_loss(v_loss)
  val_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)


EPOCHS = 5
for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    val_loss.reset_states()
    val_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for val_images, val_labels in val_ds:
        val_step(val_images, val_labels)

    template = 'Epoch: {}, train_loss: {:.5f}, train_acc: {:.2f}%, val_loss: {:.5f}, val_acc: {:.2f}%'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          val_loss.result(),
                          val_accuracy.result() * 100))

template = 'Epoch: {}, test_loss: {:.5f}, test_acc: {:.2f}%'
for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)
print(template.format(test_loss.result(),
                      test_accuracy.result() * 100))

predictions = model(tf.reshape(x_test[10], (1, 20, 38, 1)  ))
print(predictions)
print(tf.reshape(x_test[0], (1, 20, 38, 1)  ).shape)

