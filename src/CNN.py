import tensorflow as tf
import numpy as np
import util

import matplotlib.pyplot as plt
import keras
from keras.layers import Input, Dense, Activation, GlobalAveragePooling2D
from keras.models import Sequential
from keras.models import Model
from sklearn.utils import shuffle

filename = 'DenseNet201.h5'
scale = 3
X_train, X_test, y_train, y_test = util.load_data()
X_train = np.array([np.stack((np.kron(img, np.ones((scale,scale))),)*3, axis=-1) for img in X_train])
X_test = np.array([np.stack((np.kron(img, np.ones((scale,scale))),)*3, axis=-1) for img in X_test])
num_hidden = 1000
X_shape = X_train.shape
print("X_shape: ", X_shape, ' Image size: ', X_train[0,:,:,:].shape)

input_shape = (X_shape[1], X_shape[2], X_shape[3])
inputs = Input(shape=input_shape)
# keras.applications.mobilenet_v2.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet', input_tensor=inputs, classes=num_hidden)
base_model = keras.applications.densenet.DenseNet201(include_top=False, weights='imagenet', input_tensor=inputs, input_shape=input_shape, pooling=None, classes=num_hidden)
out = base_model.output
out = GlobalAveragePooling2D()(out)
out = Dense(256, activation='relu')(out)
predictions = Dense(1, activation='relu')(out)
new_model = Model(inputs=base_model.input, outputs=predictions)
new_model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

def generate_data(X_train, y_train):
    num_data = X_train.shape[0]
    batch_size = 32
    num_batch = int(num_data/batch_size)
    while True:
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        for i in range(num_batch):
            yield(X_train[i:(i+1)*batch_size,:,:,:], y_train[i:(i+1)*batch_size])

new_model.fit_generator(generate_data(X_train, y_train), steps_per_epoch=1000, epochs=10)  # starts training
y_pred = new_model.predict(X_test)
model.save_weights(filename)
print("Saved model name as ", filename)
plt.plot(y_test, y_pred)
plt.show()
