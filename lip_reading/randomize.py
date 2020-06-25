import glob
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

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
                        os.rename(frame, frame.split('/')[0] + '/' + frame.split('/')[1] + '/'
                                  + frame.split('/')[2] + '/' + str(i + 1).zfill(3) + '_' + frame.split('/')[-1])

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
                if ('noised' in images[i] or 'rand_contr' in images[i]
                        or 'vert_flip' in images[i] or 'hor_flip' in images[i]):
                    train_image.append(images[i].split('/')[3])
                    train_class.append(self.classes_dict[images[i].split('/')[3].split(']')[-1][:2]])
                else:
                    train_image.append(images[i].split('/')[3])
                    train_class.append(self.classes_dict[images[i].split('/')[3].split('_')[1]])

        train_data = pd.DataFrame()
        train_data['image'] = train_image
        train_data['class'] = train_class

        train_data.to_csv('data/' + mode + '_new.csv', header=True, index=False)

        train = pd.read_csv('data/' + mode + '_new.csv')
        print(train.head())
        print((train.tail()))
