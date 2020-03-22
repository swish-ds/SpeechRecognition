import os
import dlib
import glob
import random as rn
import math
import csv
import gc
import time
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
from numba import cuda
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import ZeroPadding3D, Conv3D, BatchNormalization, Activation, MaxPooling3D, TimeDistributed, Flatten, Bidirectional, GRU, Dense
from tensorflow.keras.models import Sequential
from keras_video_datagen import ImageDataGenerator

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

def set_record():
    record = {}
    record['loss'] = []
    record['accuracy'] = []
    record['val_loss'] = []
    record['val_accuracy'] = []

    record['loss'] += history.history['loss']
    record['accuracy'] += history.history['accuracy']
    record['val_loss'] += history.history['val_loss']
    record['val_accuracy'] += history.history['val_accuracy']

    return record

def play_sound(duration = 0.1, freq = 310):
    for time in range(10):
        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'e-' + a.split('E')[1][-1]

def create_model():
    model_small = Sequential()
    # block 1
    model_small.add(ZeroPadding3D(padding=(1, 2, 2), input_shape=(13, 70, 140, 3)))
    model_small.add(Conv3D(32, (3, 5, 5), strides=(1, 2, 2), use_bias=False))
    model_small.add(BatchNormalization(momentum=0.99))
    model_small.add(Activation('relu'))
    model_small.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    # block 2
    model_small.add(ZeroPadding3D(padding=(1, 2, 2)))
    model_small.add(Conv3D(64, (3, 5, 5), strides=(1, 1, 1), use_bias=False))
    model_small.add(BatchNormalization(momentum=0.99))
    model_small.add(Activation('relu'))
    model_small.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    # block 3
    model_small.add(ZeroPadding3D(padding=(1, 1, 1)))
    model_small.add(Conv3D(96, (3, 3, 3), strides=(1, 2, 2), use_bias=False))
    model_small.add(BatchNormalization(momentum=0.99))
    model_small.add(Activation('relu'))
    model_small.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    # RNN block
    model_small.add(TimeDistributed(Flatten()))
    model_small.add(Bidirectional(GRU(65, activation='tanh', return_sequences=False)))
    # Outputs
    model_small.add(Dense(10, activation='softmax'))

    return model_small

datagen = ImageDataGenerator()

# lrs = [1e-5, 4e-5, 7e-5, 1e-6, 4e-6, 7e-6]
lrs = [1e-5, 4e-5, 7e-5]
moms = [0.95, 0.90, 0.85, 0.80]

rn.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

print()
train_data = datagen.flow_from_directory('data/train', target_size=(70, 140), batch_size=5, frames_per_step=13, shuffle=False, color_mode='rgb')
val_data = datagen.flow_from_directory('data/validation', target_size=(70, 140), batch_size=5, frames_per_step=13, shuffle=False, color_mode='rgb')

model = create_model()
lr = 4e-5
mom = 0.90

loss_func = keras.losses.CategoricalCrossentropy(
    from_logits=True, label_smoothing=0)
sgd = keras.optimizers.SGD(learning_rate=lr, momentum=mom, nesterov=True)

model.compile(optimizer=sgd, loss=loss_func, metrics=['accuracy'], 
                loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
print("\nLearning rate = %s, Momentum = %.2f" % (format_e(keras.backend.eval(model.optimizer.lr)), keras.backend.eval(model.optimizer.momentum)))

history = model.fit(train_data, epochs=2, steps_per_epoch=209, 
                    validation_data=val_data, validation_steps=37, shuffle=False)

cuda.close()