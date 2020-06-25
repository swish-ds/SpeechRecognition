import csv
import os
import random as rn
import time
from math import ceil

import numpy as np
import tensorflow as tf
from numba import cuda
from tensorflow.keras.backend import eval
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD

import global_params
from keras_video_datagen import ImageDataGenerator
from models import LipNet, LipNetNorm


class SaveModelCallback(Callback):
    def __init__(self, lnfit):
        super(SaveModelCallback, self).__init__()
        self.model_type = lnfit.model_type
        self.val_losses = []
        self.best_val_loss = None

    def on_epoch_end(self, epoch, logs=None):
        self.val_losses.append(logs['val_loss'])
        if len(self.val_losses) > 1 and logs['val_loss'] < self.best_val_loss:
            self.best_val_loss = logs['val_loss']
            self.model.save("saved_models/%s/%s_%s_%s_%.4f_%.4f" % (
                self.model_type,
                self.model_type,
                LnFit.format_e(eval(self.model.optimizer.lr)),
                str(epoch + 1).zfill(3),
                self.best_val_loss,
                logs['val_accuracy']))
            print('\nNew best val_loss:', self.best_val_loss)
        elif len(self.val_losses) == 1:
            self.best_val_loss = self.val_losses[-1]
            self.model.save("saved_models/%s/%s_%s_%s_%.4f_%.4f" % (
                self.model_type,
                self.model_type,
                LnFit.format_e(eval(self.model.optimizer.lr)),
                str(epoch + 1).zfill(3),
                self.best_val_loss,
                logs['val_accuracy']))
            print('\nCurrent best val_loss:', self.best_val_loss)
        else:
            print('\nStill best val_loss:', self.best_val_loss)


class ResetStatesCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.reset_states()


class LnFit:
    def __init__(self, model_type, optimizer, epochs, lr=1e-4, mom=0.9, batch_s=10, classes_n=10, dropout_s=0.5,
                 frames_n=22, img_w=140, img_h=70, img_c=3):
        self.model_type = model_type
        self.optimizer = optimizer
        self.epochs = epochs
        self.lr = lr
        self.mom = mom
        self.batch_s = batch_s
        self.classes_n = classes_n
        self.dropout_s = dropout_s
        self.frames_n = frames_n
        self.img_w = img_w
        self.img_h = img_h
        self.img_c = img_c

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

    @staticmethod
    def format_e(n):
        a = '%E' % n
        return a.split('E')[0].rstrip('0').rstrip('.') + 'e-' + a.split('E')[1][-1]

    def create_csv(self):
        csv_file = "record_%s_%.4f_%.4f_%.4f_%.4f_%.4f.csv" % (
            self.format_e(eval(ln.optimizer.lr)),
            history.history['loss'][-1], history.history['accuracy'][-1],
            history.history['val_loss'][-1], history.history['val_accuracy'][-1],
            max(history.history['val_accuracy']))
        return csv_file

    def train_seq(self):
        global history, ln

        rn.seed(global_params.rn_seed)
        np.random.seed(global_params.np_random_seed)
        tf.random.set_seed(global_params.tf_random)

        print()
        datagen = ImageDataGenerator()
        train_data = datagen.flow_from_directory('data/train', target_size=(self.img_h, self.img_w),
                                                 batch_size=self.batch_s,
                                                 frames_per_step=self.frames_n, shuffle=True, seed=0,
                                                 color_mode='rgb')
        val_data = datagen.flow_from_directory('data/validation', target_size=(self.img_h, self.img_w),
                                               batch_size=self.batch_s,
                                               frames_per_step=self.frames_n, shuffle=False, seed=None,
                                               color_mode='rgb')

        if self.model_type == 'norm':
            ln = LipNetNorm(batch_s=self.batch_s, frames_n=self.frames_n, img_h=self.img_h, img_w=self.img_w,
                            img_c=self.img_c, dropout_s=self.dropout_s, output_size=self.classes_n)
        elif self.model_type == 'ln':
            ln = LipNet(batch_s=self.batch_s, frames_n=self.frames_n, img_h=self.img_h, img_w=self.img_w,
                        img_c=self.img_c, dropout_s=self.dropout_s, output_size=self.classes_n)

        ln.model().summary()

        loss_func = CategoricalCrossentropy(from_logits=True, label_smoothing=0)

        opt = None
        if self.optimizer == 'sgd':
            opt = SGD(learning_rate=self.lr, momentum=self.mom, nesterov=True)
        elif self.optimizer == 'ada':
            opt = Adam(learning_rate=self.lr)

        ln.compile(optimizer=opt, loss=loss_func, metrics=['accuracy'])

        print("\nLearning rate = %s" % (self.format_e(eval(ln.optimizer.lr))))

        steps_per_epoch = ceil(train_data.samples / (self.batch_s * self.frames_n))
        validation_steps = ceil(val_data.samples / (self.batch_s * self.frames_n))

        filepath = "saved_models/%s/%s_%s_{epoch:03d}_{val_loss:.4f}_{val_accuracy:.4f}.h5" % (
            self.model_type,
            self.model_type,
            self.format_e(eval(ln.optimizer.lr)))

        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                                     save_weights_only=True, save_best_only=True, mode='auto', save_freq='epoch')

        history = ln.fit(train_data, epochs=self.epochs, steps_per_epoch=steps_per_epoch,
                         validation_data=val_data, validation_steps=validation_steps, shuffle=False,
                         callbacks=[ResetStatesCallback(), checkpoint, SaveModelCallback(self)])

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
        # os.remove(csv_file)

        print('\nSaved into: %s\nMax accuracy: %.4f%%' % (csv_file, max(history.history['val_accuracy']) * 100))

        cuda.current_context().reset()
        tf.keras.backend.clear_session()

        self.play_sound()

        time.sleep(5)

# cuda.select_device(0)
# cuda.close()
