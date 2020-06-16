import numpy as np
import cv2
import glob
import padding
import os

base_dir = 'data'
train_dir = os.path.join(base_dir, 'train/')
val_dir = os.path.join(base_dir, 'validation/')
test_dir = os.path.join(base_dir, 'test/')

classes = 'Begin, Choose, Connection, Navigation, Next, Previous, Start, Stop, Hello, Web'
classes = 'Begin, Web'
classes = classes.split(', ')
classes_num = ['0' + str(i) if i < 10 else str(i) for i in range(1, len(classes)+1)]
classes_dict = dict(zip(classes_num, classes))

padder = padding.Pad(train_dir, val_dir, classes)
padder.pad_frames()



# # pad train
#
# for classi in classes_dict.keys():
#     for vid in vids_and_frames.keys():
#         if classi == vid.split('_')[0]:
#             if int(vids_and_frames[vid]) < target_frame_num:
#                 for i in range(1, (target_frame_num - int(vids_and_frames[vid]) + 1)):
#                     cv2.imwrite('data/train/' + classes_dict[classi] + '/' + vid + '_' + 'color' + '_' + str(int(vids_and_frames[vid]) + i).zfill(3) + '.jpg', img_black)
#
# # pad val
#
# for classi in classes_dict.keys():
#     for vid in vids_and_frames_val.keys():
#         if classi == vid.split('_')[0]:
#             if int(vids_and_frames_val[vid]) < target_frame_num:
#                 for i in range(1, (target_frame_num - int(vids_and_frames_val[vid]) + 1)):
#                     cv2.imwrite('data/validation/' + classes_dict[classi] + '/' + vid + '_' + 'color' + '_' + str(int(vids_and_frames_val[vid]) + i).zfill(3) + '.jpg', img_black)