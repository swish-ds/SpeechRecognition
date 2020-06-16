import csv
import os
import random as rn
import time
from math import ceil

import numpy as np
import tensorflow as tf
from numba import cuda
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from keras_video_datagen import ImageDataGenerator

# from tensorflow.keras.utils.plot_model

def create_model2(activation_conv):
    model_small = Sequential()
    # block 1
    model_small.add(ZeroPadding3D(input_shape=(frames_n, 70, 140, 3), padding=(1, 2, 2)))
    model_small.add(Conv3D(filters=32, kernel_size=(3, 5, 5), strides=(1, 2, 2),
                           activation=activation_conv, padding='valid', use_bias=False))
    model_small.add(BatchNormalization(momentum=0.99))
    model_small.add(SpatialDropout3D(0.5))
    model_small.add(MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    # block 2
    model_small.add(ZeroPadding3D(padding=(1, 2, 2)))
    model_small.add(Conv3D(filters=64, kernel_size=(3, 5, 5), strides=(1, 1, 1),
                           activation=activation_conv, padding='valid', use_bias=False))
    model_small.add(BatchNormalization(momentum=0.99))
    model_small.add(SpatialDropout3D(0.5))
    model_small.add(MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    # block 3
    model_small.add(ZeroPadding3D(padding=(1, 1, 1)))
    model_small.add(Conv3D(filters=96, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                           activation=activation_conv, padding='valid', use_bias=False))
    model_small.add(BatchNormalization(momentum=0.99))
    model_small.add(SpatialDropout3D(0.5))
    model_small.add(MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    # RNN block
    model_small.add(TimeDistributed(Flatten()))
    model_small.add(Bidirectional(
        GRU(256, activation='tanh', stateful=False, return_sequences=True, dropout=0, recurrent_dropout=0)))
    model_small.add(Bidirectional(
        GRU(256, activation='tanh', stateful=False, return_sequences=False, dropout=0, recurrent_dropout=0)))
    # Outputs
    model_small.add(Dense(10, activation='softmax'))

    return model_small

def create_model(activation_conv):
    model_small = Sequential()
    # block 1
    model_small.add(Conv3D(batch_input_shape=(batch_s, frames_n, 70, 140, 3),
                           filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation=activation_conv,
                           padding='valid', use_bias=False))
    model_small.add(BatchNormalization(momentum=0.99))
    model_small.add(SpatialDropout3D(0.1))
    model_small.add(MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    # block 2
    model_small.add(Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation=activation_conv,
                           padding='valid', use_bias=False))
    model_small.add(BatchNormalization(momentum=0.99))
    model_small.add(SpatialDropout3D(0.1))
    model_small.add(MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    # block 3
    model_small.add(Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation=activation_conv,
                           padding='valid', use_bias=False))
    model_small.add(BatchNormalization(momentum=0.99))
    model_small.add(SpatialDropout3D(0.1))
    model_small.add(MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    # block 4
    model_small.add(Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation=activation_conv,
                           padding='valid', use_bias=False))
    model_small.add(BatchNormalization(momentum=0.99))
    model_small.add(SpatialDropout3D(0.1))
    model_small.add(MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    # RNN block
    model_small.add(TimeDistributed(Flatten()))
    model_small.add(Bidirectional(
        GRU(100, activation='tanh', stateful=False, return_sequences=False, dropout=0,
            recurrent_dropout=0)))
    # Outputs
    model_small.add(Dense(10, activation='softmax'))

    return model_small

batch_s = 10
frames_n = 22
steps_per_epoch = ceil(28600 / (batch_s * frames_n))
validation_steps = ceil(4400 / (batch_s * frames_n))

lrs = [4e-3]
moms = [0.90]

rn.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

# datagen = ImageDataGenerator()
# train_data = datagen.flow_from_directory('data/train', target_size=(70, 140), batch_size=batch_s,
#                                          frames_per_step=frames_n, shuffle=False, color_mode='rgb')
# val_data = datagen.flow_from_directory('data/validation', target_size=(70, 140), batch_size=batch_s,
#                                        frames_per_step=frames_n, shuffle=False, color_mode='rgb')

learning_rate = 1e-4
momentum = 0.9
activation = keras.activations.relu
model = create_model(activation)

tf.keras.utils.plot_model(model, "model2.png")

loss_func = keras.losses.CategoricalCrossentropy(
    from_logits=True, label_smoothing=0)
sgd = keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd, loss=loss_func, metrics=['accuracy'])
print(model.summary())