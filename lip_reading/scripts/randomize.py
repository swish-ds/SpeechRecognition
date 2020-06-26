import glob
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import global_params

random.seed(0)
np.random.seed(0)


class Randomizer:
    def __init__(self, train_dir, val_dir, classes):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.classes = classes
        self.classes_num = ['0' + str(i) if i < 10 else str(i) for i in range(1, len(self.classes) + 1)]
        self.classes_dict = dict(zip(self.classes_num, self.classes))

    def generate_rand(self, k):
        lis = []
        while len(lis) < k:
            r = random.randint(1, k)
            if r not in lis:
                lis.append(r)
        return lis

    def shuffle(self, mode):
        print('Performing shuffle:', mode)

        frames_dir = None
        if mode == 'train':
            frames_dir = self.train_dir
        elif mode == 'val':
            frames_dir = self.val_dir

        for classi in self.classes:
            video_names = []
            frames = sorted(glob.glob(frames_dir + classi + '/' + '*.jpg'))

            for frame in frames:
                img_name = frame.split('/')[-1].split('.')[0]
                video_name = img_name.split('_color')[0]
                video_names.append(video_name)

            video_names_uniq = sorted(list(set(video_names)))
            numbers = self.generate_rand(len(video_names_uniq))
            print(numbers)

            for i in range(len(video_names_uniq)):
                video_names_uniq[i] = str(numbers[i]).zfill(3) + '_' + video_names_uniq[i]
            video_names_uniq = sorted(video_names_uniq)
            print(video_names_uniq)

            frames = sorted(glob.glob(frames_dir + classi + '/' + '*.jpg'))

            for i in range(len(video_names_uniq)):
                for frame in frames:
                    if video_names_uniq[i].split(str(i + 1).zfill(3) + '_')[1] == \
                            frame.split('/')[-1].split('.')[0].split('_color')[0]:
                        os.rename(frame,
                                  global_params.repo_dir + frame.split('/')[6] + '/' + frame.split('/')[7] + '/' +
                                  frame.split('/')[8] + '/' + frame.split('/')[9] + '/'
                                  + str(i + 1).zfill(3) + '_' + frame.split('/')[-1])

    def save_to_csv(self, mode):
        print('Saving to .csv:', mode)

        frames_dir = None
        if mode == 'train':
            frames_dir = self.train_dir
        elif mode == 'val':
            frames_dir = self.val_dir

        train_image = []
        train_class = []

        for class_id in tqdm(range(len(self.classes))):
            images = sorted(glob.glob(frames_dir + self.classes[class_id] + '/*.jpg'))
            for i in range(len(images)):
                train_image.append(images[i].split('/')[-1])
                train_class.append(self.classes_dict[images[i].split('/')[-1].split('_')[1]])

        train_data = pd.DataFrame()
        train_data['image'] = train_image
        train_data['class'] = train_class

        train_data.to_csv(global_params.base_dir + mode + '_new.csv', header=True, index=False)

        train = pd.read_csv(os.path.join(global_params.repo_dir, 'lip_reading/data/' + mode + '_new.csv'))
        print(train.head())
        print((train.tail()))
