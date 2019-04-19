import tensorflow as tf
import numpy as np
import util

import os
import ast

import matplotlib.pyplot as plt
import keras
from keras.layers import Input, Dense, Activation, Flatten, GlobalAveragePooling2D
from keras.models import Sequential
from keras.models import Model
from sklearn.utils import shuffle

data_path = "../data/simulations/data/"
data_name = "data_radius_train"
test_name = "data_radius_test.npy"
scale = 3
num_hidden = 1000
img_size_ori = (27, 10)
img_size = (36, 81, 3)
test_data = np.load(data_path + test_name)
X_test = np.array(test_data[:,0])
X_test = np.stack(X_test)
y_test = np.array(test_data[:,1])
y_test = np.stack(y_test)
# X_train, X_test, y_train, y_test = util.load_data()
# X_train = np.array([np.stack((np.kron(img, np.ones((scale,scale))),)*3, axis=-1) for img in X_train])
# X_test = np.array([np.stack((np.kron(img, np.ones((scale,scale))),)*3, axis=-1) for img in X_test])
# X_shape = X_train.shape
# print("X_shape: ", X_shape, ' Image size: ', X_train[0,:,:,:].shape)
# input_shape = (X_shape[1], X_shape[2], X_shape[3])

def mlp_basic(inputs, hidden=[1024,256,32]):
    out = Flatten()(inputs)
    for size in hidden:
        out = Dense(size, activation='relu')(out)
    out = Dense(1, activation='relu')(out)
    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])
    return model

def dense_net(inputs):
    base_model = keras.applications.densenet.DenseNet201(include_top=False, weights='imagenet', input_tensor=inputs, input_shape=input_shape, pooling=None, classes=num_hidden)
    out = base_model.output
    out = GlobalAveragePooling2D()(out)
    out = Dense(256, activation='relu')(out)
    predictions = Dense(1, activation='relu')(out)
    new_model = Model(inputs=base_model.input, outputs=predictions)
    new_model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])
    return new_model

input_shape = img_size_ori
inputs = Input(shape=input_shape)
model = mlp_basic(inputs)
filename = 'mlp_basic.h5'

def generate_data(data_path, fname, scale=3, encode_label=None, batch_size=32, mode="train"):
    filename_list = os.listdir(data_path)
    filename_list = [name for name in filename_list if (fname in name)]
    num_files = len(filename_list)
    file_idx = 0
    f = open(data_path+filename_list[file_idx], 'r')
    while True:
        images = []
        labels = []
        while len(images) < batch_size:
            if file_idx >= num_files-1:
                # Reset file index
                file_idx = -1
            line = f.readline()
            # check to see if the line is empty, indicating we have
            # reached the end of the file
            if line == "":
                # Read the next file
                file_idx += 1
                f = open(data_path+filename_list[file_idx], 'r')
                line = f.readline()

            # if we are evaluating we should now break from our
            # loop to ensure we don't continue to fill up the
            # batch from samples at the beginning of the file
            if mode == "eval":
                break
                # extract the label and construct the image
            line = line.strip().split(";")
            label = line[0]
            img_line = line[1]
            image = np.array(ast.literal_eval(img_line))
            # image = np.stack((np.kron(image, np.ones((scale,scale))),)*3, axis=-1)
            images.append(image)
            labels.append(label)
            # one-hot encode the labels
            if encode_label is not None:
                label = encode_label.transform(label)

        images = np.array(images)
        labels = np.array(labels)
        # yield the batch to the calling function
        yield (images, labels)

# cnt = 0
# data_gen = generate_data(data_path, data_name)
# while cnt < 1:
#     img, label = next(data_gen)
#     print('img: ', img[0])
#     plt.cla()
#     plt.imshow(img[0])
#     plt.show()
#     cnt += 1

# model.fit_generator(generate_data(data_path, data_name), steps_per_epoch=1000, epochs=10)  # starts training
model.load_weights(filename)
y_pred = model.predict(X_test)
# model.save_weights(filename)
# print("Saved model name as ", filename)
plt.scatter(y_test, y_pred)
plt.title("MLP - Predicted vs. Actual Test Radius")
plt.xlabel("Actual Radius (nm)")
plt.ylabel("Predicted Radius (nm)")
plt.show()
