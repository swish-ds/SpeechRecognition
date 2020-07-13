import glob
import os

import cv2
from tqdm import tqdm
import numpy as np
import imutils
from imutils import face_utils


class Extractor:
    def __init__(self, train_dir, val_dir, test_dir, classes, detector, predictor, scale,
                 train_people, val_people, size_x, size_y, test_people=None):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.classes = classes
        self.detector = detector
        self.predictor = predictor
        self.scale = scale
        self.train_people = train_people
        self.val_people = val_people
        self.test_people = test_people
        self.size_x = size_x
        self.size_y = size_y

        self.classes_num = ['0' + str(i) if i < 10 else str(i) for i in range(1, len(classes) + 1)]
        self.word_ids = ['0' + str(i) if i < 10 else str(i) for i in range(1, 11)]
        self.classes_dict = dict(zip(self.classes_num, classes))

    def create_dirs(self):
        """ Create directories ./data/train, ./data/val, ./data/test """
        print('\nCreating directories:')
        for class_name in self.classes:
            train_vids_dir = os.path.join(self.train_dir, class_name)
            val_vids_dir = os.path.join(self.val_dir, class_name)
            test_vids_dir = os.path.join(self.test_dir, class_name)

            try:
                os.makedirs(train_vids_dir)
            except IOError as e:
                print('Error: ', e)

            try:
                os.makedirs(val_vids_dir)
            except IOError as e:
                print('Error: ', e)

            try:
                os.makedirs(test_vids_dir)
            except IOError as e:
                print('Error: ', e)

            print(class_name, 'directory created')

    def remove_dirs(self):
        """ Remove directories ./data/train, ./data/val, ./data/test """
        print('\nRemoving directories:')
        try:
            for root, dirs, files in os.walk(self.train_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.train_dir)
            print('Train directory removed')
        except IOError as e:
            print('Error: ', e)

        try:
            for root, dirs, files in os.walk(self.val_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.val_dir)
            print('Val directory removed')
        except IOError as e:
            print('Error: ', e)

        try:
            for root, dirs, files in os.walk(self.test_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.test_dir)
            print('Test directory removed')
        except IOError as e:
            print('Error: ', e)

    def extract_crop(self, mode):
        people_list = None
        if mode == 'train':
            people_list = self.train_people
        elif mode == 'val':
            people_list = self.val_people
        elif mode == 'test':
            people_list = self.test_people

        if (mode == 'train' and os.path.isdir(self.train_dir)) \
                or (mode == 'val' and os.path.isdir(self.val_dir)) \
                or (mode == 'test' and os.path.isdir(self.test_dir)):
            print('\nExtract and crop:', mode)

            for classi in self.classes_num[:1]:
                for person in people_list[:1]:
                    for word_id in self.word_ids[:1]:
                        for f in sorted(
                                glob.glob(
                                    os.path.join('data/miracl/' + person + '/words/' + classi + '/' + word_id,
                                                 "*.jpg"))):
                            print(f)
                            img = cv2.imread(f, 1)
                            # print(img.shape)
                            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            faces = self.detector(img)
                            # print(faces)
                            if faces:
                                faces =  [faces[0]]

                            for k, faces in enumerate(faces):
                                # x1 = faces.left()
                                # y1 = faces.top()
                                # x2 = faces.right()
                                # y2 = faces.bottom()

                                # cv2.circle(img=img, center=(x1, y1), radius=2, color=(255, 0, 0), thickness=6)
                                # cv2.circle(img=img, center=(x2, y2), radius=2, color=(255, 0, 0), thickness=6)
                                # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))

                                shape = self.predictor(img, faces)
                                # shape = face_utils.shape_to_np(shape)

                                x_51 = shape.part(51).x
                                y_51 = shape.part(51).y
                                x_57 = shape.part(57).x
                                y_57 = shape.part(57).y
                                x_48 = shape.part(48).x
                                y_48 = shape.part(48).y
                                x_54 = shape.part(54).x
                                y_54 = shape.part(54).y

                                x_62 = shape.part(62).x
                                y_62 = shape.part(62).y
                                x_66 = shape.part(66).x
                                y_66 = shape.part(66).y

                                x_50 = shape.part(50).x
                                y_50 = shape.part(50).y
                                x_52 = shape.part(52).x
                                y_52 = shape.part(52).y

                                x1_m = x_51 - 18
                                y1_m = y_51 - 7
                                x2_m = x_57 + 18
                                y2_m = y_57 + 9

                                # print(y_51, y_57, (y_57 - y_51) * 0.2, (y_57 - y_51) * 0.1)
                                # print(x_48, x_54, (x_54 - x_48) * 0.2, (x_54 - x_48) * 0.1)

                                # For the Russian dataset
                                # y1_m = int(y_51 - ((y_57 - y_51) * 0.3))
                                # y2_m = int(y_57 + ((y_57 - y_51) * 0.1))
                                # x1_m = int(x_48 - ((x_54 - x_48) * 0.1))
                                # x2_m = int(x_54 + ((x_54 - x_48) * 0.1))

                                cv2.circle(img=img, center=(x_51, y_51), radius=2, color=(0, 255, 0), thickness=1)
                                cv2.circle(img=img, center=(x_57, y_57), radius=2, color=(0, 255, 0), thickness=1)
                                cv2.circle(img=img, center=(x_48, y_48), radius=2, color=(0, 255, 0), thickness=1)
                                cv2.circle(img=img, center=(x_54, y_54), radius=2, color=(0, 255, 0), thickness=1)
                                # cv2.circle(img=img, center=(x_48, y_51), radius=2, color=(0, 0, 255), thickness=1)
                                # cv2.circle(img=img, center=(x_54, y_57), radius=2, color=(0, 0, 255), thickness=1)

                                # cv2.circle(img=img, center=(x_62, y_62), radius=2, color=(0, 0, 255), thickness=1)
                                # cv2.circle(img=img, center=(x_66, y_66), radius=2, color=(0, 0, 255), thickness=1)
                                #
                                # cv2.circle(img=img, center=(x_62, y_62), radius=2, color=(0, 0, 255), thickness=1)
                                # cv2.circle(img=img, center=(x_66, y_66), radius=2, color=(0, 0, 255), thickness=1)

                                # img = img[y1_m:y2_m, x1_m:x2_m]
                                # img = cv2.resize(img, (140, 70))


                                offset_x_m = (70 - (abs(x1_m - x2_m))) / 2
                                offset_y_m = (35 - (abs(y1_m - y2_m))) / 2

                                x1_m2 = x1_m - int(offset_x_m)
                                y1_m2 = y1_m - int(offset_y_m)
                                x2_m2 = x2_m + int(offset_x_m)
                                y2_m2 = y2_m + int(offset_y_m)

                                cv2.rectangle(img, (x1_m2, y1_m2), (x2_m2, y2_m2), (0, 0, 255))

                                img = img[int(y1_m - offset_y_m - 10):int(y2_m + offset_y_m + 10),
                                          int(x1_m - offset_x_m - 10):int(x2_m + offset_x_m + 10)]


                                # For the Russian dataset
                                # img = img[int(y1_m):int(y2_m), int(x1_m):int(x2_m)]
                                img = cv2.resize(img, (400, 200),
                                                 interpolation=cv2.INTER_AREA)

                                # img = cv2.resize(img, (int(img.shape[1] * 200 / 100),
                                #                        int(img.shape[0] * 200 / 100)),
                                #                  interpolation=cv2.INTER_AREA)


                            if mode == 'train':
                                cv2.imwrite(
                                    self.train_dir + self.classes_dict[classi] + '/' + classi + '_' + person + '_'
                                    + word_id + '_' + f[28:-4] + '.jpg', img)
                            elif mode == 'val':
                                cv2.imwrite(
                                    self.val_dir + self.classes_dict[classi] + '/' + classi + '_' + person + '_'
                                    + word_id + '_' + f[28:-4] + '.jpg', img)
                            elif mode == 'test':
                                cv2.imwrite(
                                    self.test_dir + self.classes_dict[classi] + '/' + classi + '_' + person + '_'
                                    + word_id + '_' + f[28:-4] + '.jpg', img)
        else:
            print('\n', mode, 'directory does not exist')
