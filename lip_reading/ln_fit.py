import csv
import os
import time
import random as rn
import numpy as np
from math import ceil
from numba import cuda

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.backend import eval
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import CategoricalCrossentropy

from keras_video_datagen import ImageDataGenerator

from models import LipNet, LipNetNorm


class ResetStatesCallback(Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()


class LnFit:
    def __init__(self, model_type, optimizer, epochs, lr=1e-4, mom=0.9, batch_s=10, classes_n=10, frames_n=22, img_w=140, img_h=70, img_c=3):
        self.model_type = model_type
        self.optimizer = optimizer
        self.epochs = epochs
        self.lr = lr
        self.mom = mom
        self.batch_s = batch_s
        self.classes_n = classes_n
        self.frames_n = frames_n
        self.img_w = img_w
        self.img_h = img_h
        self.img_c = img_c

        self.steps_per_epoch = ceil(28600 / (self.batch_s * self.frames_n))
        self.validation_steps = ceil(4400 / (self.batch_s * self.frames_n))

        gpus = tf.config.experimental.list_physical_devices('GPU')

        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)

    def set_record(self):
        record = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        record['loss'] += history.history['loss']
        record['accuracy'] += history.history['accuracy']
        record['val_loss'] += history.history['val_loss']
        record['val_accuracy'] += history.history['val_accuracy']

        return record

    def play_sound(self, duration=0.1, freq=310):
        for time in range(10):
            os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

    def format_e(self, n):
        a = '%E' % n
        return a.split('E')[0].rstrip('0').rstrip('.') + 'e-' + a.split('E')[1][-1]

    # LipNetNorm
    def create_model2(self, ):
        model_small = Sequential()
        # block 1
        model_small.add(ZeroPadding3D(input_shape=(self.frames_n, 70, 140, 3), padding=(1, 2, 2)))
        model_small.add(Conv3D(filters=32, kernel_size=(3, 5, 5), strides=(1, 2, 2),
                               activation='relu', padding='valid', use_bias=False))
        model_small.add(BatchNormalization(momentum=0.99))
        model_small.add(SpatialDropout3D(0.5))
        model_small.add(MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

        # block 2
        model_small.add(ZeroPadding3D(padding=(1, 2, 2)))
        model_small.add(Conv3D(filters=64, kernel_size=(3, 5, 5), strides=(1, 1, 1),
                               activation='relu', padding='valid', use_bias=False))
        model_small.add(BatchNormalization(momentum=0.99))
        model_small.add(SpatialDropout3D(0.5))
        model_small.add(MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

        # block 3
        model_small.add(ZeroPadding3D(padding=(1, 1, 1)))
        model_small.add(Conv3D(filters=96, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                               activation='relu', padding='valid', use_bias=False))
        model_small.add(BatchNormalization(momentum=0.99))
        model_small.add(SpatialDropout3D(0.5))
        model_small.add(MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

        # Reshape
        model_small.add(TimeDistributed(Flatten()))

        # RNN block
        model_small.add(Bidirectional(
            GRU(256, activation='tanh', stateful=False, return_sequences=True,
                dropout=0, recurrent_dropout=0)))
        model_small.add(Bidirectional(
            GRU(256, activation='tanh', stateful=False, return_sequences=False,
                dropout=0, recurrent_dropout=0)))
        # Outputs
        model_small.add(Dense(10, activation='softmax'))

        return model_small

    def create_csv2(self):
        csv_file = "record_%s_%.2f_%.4f_%.4f_%.4f_%.4f_%.4f.csv" % (
            self.format_e(eval(ln.optimizer.lr)), eval(ln.optimizer.momentum),
            history.history['loss'][-1], history.history['accuracy'][-1],
            history.history['val_loss'][-1], history.history['val_accuracy'][-1],
            max(history.history['val_accuracy']))
        return csv_file

    def create_csv(self):
        csv_file = "record_%s_%.4f_%.4f_%.4f_%.4f_%.4f.csv" % (
            self.format_e(eval(ln.optimizer.lr)),
            history.history['loss'][-1], history.history['accuracy'][-1],
            history.history['val_loss'][-1], history.history['val_accuracy'][-1],
            max(history.history['val_accuracy']))
        return csv_file

    def train_seq(self):
        global history, ln

        rn.seed(0)
        np.random.seed(0)
        tf.random.set_seed(0)

        print()
        datagen = ImageDataGenerator()
        train_data = datagen.flow_from_directory('data/train', target_size=(70, 140), batch_size=self.batch_s,
                                                 frames_per_step=self.frames_n, shuffle=False, color_mode='rgb')
        val_data = datagen.flow_from_directory('data/validation', target_size=(70, 140), batch_size=self.batch_s,
                                               frames_per_step=self.frames_n, shuffle=False, color_mode='rgb')

        if self.model_type == 'norm':
            ln = LipNetNorm(batch_s=self.batch_s, frames_n=self.frames_n, img_h=self.img_h, img_w=self.img_w,
                        img_c=self.img_c, output_size=self.classes_n)
        elif self.model_type == 'ln':
            ln = LipNet(batch_s=self.batch_s, frames_n=self.frames_n, img_h=self.img_h, img_w=self.img_w,
                        img_c=self.img_c, output_size=self.classes_n)

        # print(ln.model().summary())

        loss_func = CategoricalCrossentropy(from_logits=True, label_smoothing=0)

        opt = None
        if self.optimizer == 'sgd':
            opt = SGD(learning_rate=self.lr, momentum=self.mom, nesterov=True)
        elif self.optimizer == 'ada':
            opt = Adam(learning_rate=self.lr)

        ln.compile(optimizer=opt, loss=loss_func, metrics=['accuracy'])

        print("\nLearning rate = %s" % (self.format_e(eval(ln.optimizer.lr))))

        history = ln.fit(train_data, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch,
                         validation_data=val_data, validation_steps=self.validation_steps, shuffle=False,
                         callbacks=[ResetStatesCallback()])

        csv_file = self.create_csv()

        record = self.set_record()
        with open(csv_file, 'w') as f:
            w = csv.writer(f)
            w.writerow(record.keys())
            for i in range(len(record['loss'])):
                a = []
                for idx in range(len(record)):
                    a.append(list(record.values())[idx][i])
                w.writerow(a)
        os.remove(csv_file)

        print('\nSaved into: %s\nMax accuracy: %.4f%%' % (csv_file, max(history.history['val_accuracy']) * 100))

        cuda.current_context().reset()
        tf.keras.backend.clear_session()

        self.play_sound()

        time.sleep(2)

# cuda.select_device(0)
# cuda.close()
