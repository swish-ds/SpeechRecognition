import tensorflow as tf
from utils import global_params
import numpy as np
import random as rn
from tensorflow.keras.layers import Conv3D, ZeroPadding3D, MaxPool3D, Dense, Flatten, GRU, Bidirectional, \
    SpatialDropout3D, BatchNormalization, TimeDistributed, Input
from tensorflow.keras.models import Model

rn.seed(global_params.rn_seed)
np.random.seed(global_params.np_random_seed)
tf.random.set_seed(global_params.tf_random)


# padding=(1, 0, 0), flatten
class LipNetNorm2(tf.keras.Model):
    def __init__(self, batch_s=10, frames_n=20, img_h=70, img_w=140, img_c=3, dropout_s=0.5, output_size=10):
        super(LipNetNorm2, self).__init__()
        self.batch_s = batch_s
        self.frames_n = frames_n
        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        self.dropout_s = dropout_s
        self.output_size = output_size

        # block 1
        self.zero1 = ZeroPadding3D(padding=(1, 0, 0))
        self.conv1 = Conv3D(filters=32, kernel_size=(3, 5, 5), strides=(1, 2, 2),
                            activation='relu', padding='valid', use_bias=False)
        self.norm1 = BatchNormalization(momentum=0.99)
        self.drop1 = SpatialDropout3D(self.dropout_s)
        self.maxp1 = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))

        # block 2
        self.zero2 = ZeroPadding3D(padding=(1, 0, 0))
        self.conv2 = Conv3D(filters=64, kernel_size=(3, 5, 5), strides=(1, 1, 1),
                            activation='relu', padding='valid', use_bias=False)
        self.norm2 = BatchNormalization(momentum=0.99)
        self.drop2 = SpatialDropout3D(self.dropout_s)
        self.maxp2 = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))

        # block 3
        self.zero3 = ZeroPadding3D(padding=(1, 0, 0))
        self.conv3 = Conv3D(filters=96, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                            activation='relu', padding='valid', use_bias=False)
        self.norm3 = BatchNormalization(momentum=0.99)
        self.drop3 = SpatialDropout3D(self.dropout_s)
        self.maxp3 = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))

        # Reshape
        self.resh1 = TimeDistributed(Flatten())

        # RNN block
        self.gru1 = Bidirectional(
            GRU(256, activation='tanh', stateful=False, return_sequences=True,
                dropout=0, recurrent_dropout=0))
        self.gru2 = Bidirectional(
            GRU(256, activation='tanh', stateful=False, return_sequences=False,
                dropout=0, recurrent_dropout=0))
        # Outputs
        self.out = Dense(self.output_size, activation='softmax')

    def call(self, inputs, **kwargs):
        # block 1
        x = self.zero1(inputs)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.drop1(x)
        x = self.maxp1(x)
        # block2
        x = self.zero2(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.drop2(x)
        x = self.maxp2(x)
        # block 3
        x = self.zero3(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.drop3(x)
        x = self.maxp3(x)
        # reshape
        x = self.resh1(x)
        # rnn block
        x = self.gru1(x)
        x = self.gru2(x)

        return self.out(x)

    def model(self):
        ins = Input(shape=(self.frames_n, self.img_h, self.img_w, self.img_c), batch_size=self.batch_s)
        return Model(inputs=ins, outputs=self.call(ins))


# padding=(1, 0, 0), GlobalAveragePooling2D
class LipNetNorm6(tf.keras.Model):
    def __init__(self, batch_s=10, frames_n=20, img_h=70, img_w=140, img_c=3, dropout_s=0.5, output_size=10):
        super(LipNetNorm6, self).__init__()
        self.batch_s = batch_s
        self.frames_n = frames_n
        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        self.dropout_s = dropout_s
        self.output_size = output_size

        # block 1
        self.zero1 = ZeroPadding3D(padding=(1, 0, 0))
        self.conv1 = Conv3D(filters=32, kernel_size=(3, 5, 5), strides=(1, 2, 2),
                            activation='relu', padding='valid', use_bias=False)
        self.norm1 = BatchNormalization(momentum=0.99)
        self.drop1 = SpatialDropout3D(self.dropout_s)
        self.maxp1 = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))

        # block 2
        self.zero2 = ZeroPadding3D(padding=(1, 0, 0))
        self.conv2 = Conv3D(filters=64, kernel_size=(3, 5, 5), strides=(1, 1, 1),
                            activation='relu', padding='valid', use_bias=False)
        self.norm2 = BatchNormalization(momentum=0.99)
        self.drop2 = SpatialDropout3D(self.dropout_s)
        self.maxp2 = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))

        # block 3
        self.zero3 = ZeroPadding3D(padding=(1, 0, 0))
        self.conv3 = Conv3D(filters=96, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                            activation='relu', padding='valid', use_bias=False)
        self.norm3 = BatchNormalization(momentum=0.99)
        self.drop3 = SpatialDropout3D(self.dropout_s)
        self.maxp3 = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))

        # Reshape
        self.resh1 = TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())

        # RNN block
        self.gru1 = Bidirectional(
            GRU(256, activation='tanh', stateful=False, return_sequences=True,
                dropout=0, recurrent_dropout=0))
        self.gru2 = Bidirectional(
            GRU(256, activation='tanh', stateful=False, return_sequences=False,
                dropout=0, recurrent_dropout=0))
        # Outputs
        self.out = Dense(self.output_size, activation='softmax')

    def call(self, inputs, **kwargs):
        # block 1
        x = self.zero1(inputs)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.drop1(x)
        x = self.maxp1(x)
        # block2
        x = self.zero2(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.drop2(x)
        x = self.maxp2(x)
        # block 3
        x = self.zero3(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.drop3(x)
        x = self.maxp3(x)
        # reshape
        x = self.resh1(x)
        # rnn block
        x = self.gru1(x)
        x = self.gru2(x)

        return self.out(x)

    def model(self):
        ins = Input(shape=(self.frames_n, self.img_h, self.img_w, self.img_c), batch_size=self.batch_s)
        return Model(inputs=ins, outputs=self.call(ins))


