import os
import random as rn
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import *
from utils import global_params
from utils.keras_video_datagen import ImageDataGenerator

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def model_predict():
    datagen = ImageDataGenerator()
    val_data = datagen.flow_from_directory(os.path.join(global_params.repo_dir[:-1], "lip_reading/data/validation"),
                                           augm=False,
                                           target_size=(global_params.img_h, global_params.img_w),
                                           batch_size=1,
                                           frames_per_step=global_params.frames_n, shuffle=False, seed=None,
                                           color_mode='rgb')
    filenames = val_data.filenames
    nb_samples = len(filenames)

    val_classes = []
    for i in list(range(0, int(len(val_data.classes)), global_params.frames_n)):
        val_classes.append(int(np.mean(np.array(val_data.classes[i:i + global_params.frames_n]))))

    target_names = global_params.classes

    ln = load_model(os.path.join(global_params.repo_dir,
                                 'lip_reading/models/results/norm/rus_augmented/LipNetNorm6/norm_1e-4_201_1.6985_0.7700'))

    # print(ln.summary())

    prediction = ln.predict(val_data, steps=ceil(nb_samples / global_params.frames_n), verbose=1)
    pred_classes = prediction.argmax(axis=-1)

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))

    array = confusion_matrix(val_classes, pred_classes)

    df_cm = pd.DataFrame(array, index=[i for i in target_names],
                         columns=[i for i in target_names])

    ax1 = sn.heatmap(df_cm, annot=True)
    fig.set_facecolor("#F6F6F6")
    # fig.suptitle((
    #     'SGD(lr=4e-3, momentum=0.90, nesterov=True)\n\
    #     loss: 1.4656 - acc: 0.9969 - val_loss: 1.8118 - val_acc: 0.6450'), fontsize=16, color='Black')
    fig.show()
    # plt.savefig('[rus_augmented_LipNetNorm6][matrix]norm_1e-4_201_1.6985_0.7700.png')

    print(classification_report(val_classes, pred_classes, target_names=target_names))

    ln.evaluate(val_data, steps=ceil(nb_samples / global_params.frames_n))


model_predict()
