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

from utils import global_params
from utils.keras_video_datagen import ImageDataGenerator
from models.lr_models import LipNetNorm2, LipNetNorm6


class SaveModelCallback(Callback):
    def __init__(self, lnfit):
        super(SaveModelCallback, self).__init__()
        self.model_type = lnfit.model_type
        self.val_losses = []
        self.best_val_loss = None
        self.best_acc = None
        self.lnfit = LnFit()
        self.model_logs = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    def set_record(self, model_logs):
        record = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        record['loss'] += model_logs['loss']
        record['accuracy'] += model_logs['accuracy']
        record['val_loss'] += model_logs['val_loss']
        record['val_accuracy'] += model_logs['val_accuracy']
        return record

    def create_csv(self, epoch):
        csv_file = os.path.join(os.path.join(global_params.repo_dir, "lip_reading/models/results/%s/%s_%s_%s_%.4f_%.4f.csv" % (
                    self.model_type,
                    self.model_type,
                    self.lnfit.format_e(self.lnfit.lr),
                    str(epoch + 1).zfill(3),
                    self.best_val_loss,
                    self.best_acc)))
        return csv_file

    def on_epoch_end(self, epoch, logs=None):
        self.model_logs['loss'] += [logs['loss']]
        self.model_logs['accuracy'] += [logs['accuracy']]
        self.model_logs['val_loss'] += [logs['val_loss']]
        self.model_logs['val_accuracy'] += [logs['val_accuracy']]
        self.val_losses.append(logs['val_loss'])

        if len(self.val_losses) > 1 and logs['val_loss'] < self.best_val_loss:
            self.best_val_loss = logs['val_loss']
            self.best_acc = logs['val_accuracy']
            if self.best_val_loss < 1.76:
                self.model.save(os.path.join(global_params.repo_dir, "lip_reading/models/results/%s/%s_%s_%s_%.4f_%.4f" % (
                    self.model_type,
                    self.model_type,
                    self.lnfit.format_e(self.lnfit.lr),
                    str(epoch + 1).zfill(3),
                    self.best_val_loss,
                    self.best_acc)))
                self.model.save_weights(os.path.join(global_params.repo_dir, "lip_reading/models/results/%s/%s_%s_%s_%.4f_%.4f.h5" % (
                    self.model_type,
                    self.model_type,
                    self.lnfit.format_e(self.lnfit.lr),
                    str(epoch + 1).zfill(3),
                    self.best_val_loss,
                    self.best_acc)))

                csv_file = self.create_csv(epoch)
                record = self.set_record(self.model_logs)
                with open(csv_file, 'w') as f:
                    w = csv.writer(f)
                    w.writerow(record.keys())
                    for i in range(len(record['loss'])):
                        a = []
                        for idx in range(len(record)):
                            a.append(list(record.values())[idx][i])
                        w.writerow(a)
                f.close()

            print('\nNew best val_loss:', self.best_val_loss)
        elif len(self.val_losses) == 1:
            self.best_val_loss = self.val_losses[-1]
            self.best_acc = logs['val_accuracy']
            print('\nCurrent best val_loss:', self.best_val_loss)
        else:
            print('\nStill best val_loss:', self.best_val_loss)


class ResetStatesCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.reset_states()


class LnFit:
    def __init__(self, model_type='norm', optimizer='ada', epochs=300, lr=1e-4, mom=0.9, batch_s=10, classes_n=10, dropout_s=0.5,
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

    def format_e(self, n):
        a = '%E' % n
        return a.split('E')[0].rstrip('0').rstrip('.') + 'e-' + a.split('E')[1][-1]

    def create_csv(self):
        csv_file = os.path.join(global_params.repo_dir, "lip_reading/models/results/%s/%s_%s_%.4f_%.4f_%.4f_%.4f_%.4f.csv" % (
            self.model_type,
            self.model_type,
            self.format_e(eval(ln.optimizer.lr)),
            history.history['loss'][-1], history.history['accuracy'][-1],
            history.history['val_loss'][-1], history.history['val_accuracy'][-1],
            max(history.history['val_accuracy'])))
        return csv_file

    def train_seq(self):
        global history, ln

        rn.seed(global_params.rn_seed)
        np.random.seed(global_params.np_random_seed)
        tf.random.set_seed(global_params.tf_random)

        print()
        datagen = ImageDataGenerator()
        train_data = datagen.flow_from_directory(os.path.join(global_params.repo_dir, "lip_reading/data/train"),
                                                 augm=False,
                                                 target_size=(self.img_h, self.img_w),
                                                 batch_size=self.batch_s,
                                                 frames_per_step=self.frames_n, shuffle=True, seed=0,
                                                 color_mode='rgb')
        val_data = datagen.flow_from_directory(os.path.join(global_params.repo_dir, "lip_reading/data/validation"),
                                               augm=False,
                                               target_size=(self.img_h, self.img_w),
                                               batch_size=self.batch_s,
                                               frames_per_step=self.frames_n, shuffle=False, seed=None,
                                               color_mode='rgb')

        if self.model_type == 'norm':
            ln = LipNetNorm6(batch_s=self.batch_s, frames_n=self.frames_n, img_h=self.img_h, img_w=self.img_w,
                             img_c=self.img_c, dropout_s=self.dropout_s, output_size=self.classes_n)
        # elif self.model_type == 'ln':
        #     ln = LipNet(batch_s=self.batch_s, frames_n=self.frames_n, img_h=self.img_h, img_w=self.img_w,
        #                 img_c=self.img_c, dropout_s=self.dropout_s, output_size=self.classes_n)

        ln.model().summary()
        # ln.model()

        loss_func = CategoricalCrossentropy(from_logits=True, label_smoothing=0)

        opt = None
        if self.optimizer == 'sgd':
            opt = SGD(learning_rate=self.lr, momentum=self.mom, nesterov=True)
        elif self.optimizer == 'ada':
            opt = Adam(learning_rate=self.lr)

        ln.compile(optimizer=opt, loss=loss_func, metrics=['accuracy'])

        print("\nLearning rate = %s" % (self.format_e(eval(ln.optimizer.lr))))

        print(train_data.samples, self.batch_s, self.frames_n)
        print(val_data.samples, self.batch_s, self.frames_n)

        steps_per_epoch = ceil(train_data.samples / (self.batch_s * self.frames_n))
        validation_steps = ceil(val_data.samples / (self.batch_s * self.frames_n))

        filepath = os.path.join(global_params.repo_dir, "lip_reading/models/results/%s/%s_%s_{epoch:03d}_{val_loss:.4f}_{val_accuracy:.4f}.h5" % (
            self.model_type,
            self.model_type,
            self.format_e(eval(ln.optimizer.lr))))

        history = ln.fit(train_data, epochs=self.epochs, steps_per_epoch=steps_per_epoch,
                         validation_data=val_data, validation_steps=validation_steps, shuffle=False,
                         callbacks=[ResetStatesCallback(), SaveModelCallback(self)])
        # history = ln.fit(train_data, epochs=self.epochs, steps_per_epoch=steps_per_epoch,
        #                  validation_data=val_data, validation_steps=validation_steps, shuffle=False,
        #                  callbacks=[ResetStatesCallback()])

        # csv_file = self.create_csv()
        #
        # record = self.set_record()
        # with open(csv_file, 'w') as f:
        #     w = csv.writer(f)
        #     w.writerow(record.keys())
        #     for i in range(len(record['loss'])):
        #         a = []
        #         for idx in range(len(record)):
        #             a.append(list(record.values())[idx][i])
        #         w.writerow(a)
        # # os.remove(csv_file)

        cuda.current_context().reset()
        tf.keras.backend.clear_session()

        self.play_sound()

        time.sleep(5)

# cuda.select_device(0)
# cuda.close()
