from utils import global_params
from tqdm import tqdm
import glob
import os
import cv2

classes = global_params.classes
classes_num = ['0' + str(i) if i < 10 else str(i) for i in range(1, len(classes) + 1)]
word_ids = ['0' + str(i) if i < 10 else str(i) for i in range(1, 11)]
classes_dict = dict(zip(classes_num, classes))
people_list = global_params.train_people + global_params.val_people

# cnt = 0
# for classi in tqdm(classes_num[:]):
#     for person in people_list[:]:
#         for word_id in word_ids[:]:
#             for f in sorted(
#                     glob.glob(
#                         os.path.join('data/miracl/' + person + '/words/' + classi + '/' + word_id, '*.mp4'))):
#                 cnt += 1
#                 # os.rename(f, 'data/miracl/' + person + '/words/' + classi + '/' + word_id + '/color.mp4')
#
#
# print(cnt)


# cnt = 0
# for classi in tqdm(classes[:]):
#     for f in sorted(
#             glob.glob(
#                 os.path.join('data/train/' + classi, '*.jpg'))):
#         if 'F11' in f:
#             print(f)
#             cnt += 1
#
# print(cnt)
