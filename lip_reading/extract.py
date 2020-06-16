import dlib
import cv2
import glob
import os
from tqdm import tqdm


class Extractor:
    def __init__(self, base_dir, train_dir, val_dir, test_dir, classes, detector, predictor, scale,
                 train_people, val_people, test_people=None):
        self.base_dir = base_dir
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

        self.classes_num = ['0' + str(i) if i < 10 else str(i) for i in range(1, len(classes)+1)]
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
            counter = 0
            for classi in tqdm(self.classes_num[:]):
                for person in people_list[:]:
                    for word_id in self.word_ids[:]:
                        for f in sorted(
                                glob.glob(
                                    os.path.join('data/miracl/' + person + '/words/' + classi + '/' + word_id, "*.jpg"))):
                            img = cv2.imread(f, 1)
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            rects = self.detector(gray)

                            for k, rect in enumerate(rects):
                                shape = self.predictor(gray, rect)

                                x_51 = shape.part(51).x
                                y_51 = shape.part(51).y
                                x_57 = shape.part(57).x
                                y_57 = shape.part(57).y

                                x1_m = x_51 - 18
                                y1_m = y_51 - 7
                                x2_m = x_57 + 18
                                y2_m = y_57 + 9

                                offset_x_m = (70 - (abs(x1_m - x2_m))) / 2
                                offset_y_m = (35 - (abs(y1_m - y2_m))) / 2

                                img = img[int(y1_m - offset_y_m):int(y2_m + offset_y_m),
                                          int(x1_m - offset_x_m):int(x2_m + offset_x_m)]

                                img = cv2.resize(img, (int(img.shape[1] * self.scale / 100),
                                                       int(img.shape[0] * self.scale / 100)),
                                                 interpolation=cv2.INTER_AREA)

                            counter += 1

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