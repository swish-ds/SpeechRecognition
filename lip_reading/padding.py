import numpy as np
import cv2
import os
import glob


class Pad:
    def __init__(self, train_dir, val_dir, classes):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.classes = classes
        self.classes_num = ['0' + str(i) if i < 10 else str(i) for i in range(1, len(self.classes)+1)]
        self.word_ids = ['0' + str(i) if i < 10 else str(i) for i in range(1, 11)]
        self.classes_dict = dict(zip(self.classes_num, self.classes))
        self.vids_and_frames = self.count_frames(mode='train')
        self.vids_and_frames_val = self.count_frames(mode='val')

    def count_frames(self, mode):
        frames_dir = None
        if mode == 'train':
            frames_dir = self.train_dir
        elif mode == 'val':
            frames_dir = self.val_dir

        all_images = []
        video_names = []
        frame_nums = []
        frame_nums_uniq = []
        vids_and_frames = {}
        frames_distribution = {}

        for classi in self.classes_dict.values():
            for i in sorted(glob.glob(frames_dir + classi + '/*.jpg')):
                all_images.append(i)

        for img in all_images:
            img_name = img.split('/')[-1].split('.')[0]
            video_name = img_name[:9]
            video_names.append(video_name)
            frame_num = img_name[-3:]
            frame_nums.append(frame_num)

        video_names_uniq = list(sorted(set(video_names)))

        for i in range(len(video_names)):
            if i < len(video_names) - 1:
                if video_names[i] == video_names[i + 1]:
                    pass
                else:
                    frame_nums_uniq.append(frame_nums[i])
            else:
                frame_nums_uniq.append(frame_nums[-1])

        for i in range(len(video_names_uniq)):
            vids_and_frames[video_names_uniq[i]] = frame_nums_uniq[i]

        for frame_num_uniq in sorted(set(frame_nums_uniq)):
            count_d = 0
            for vid_name, frames_num in list(vids_and_frames.items()):
                if frames_num == frame_num_uniq:
                    count_d += 1
                    frames_distribution[frame_num_uniq] = count_d

        print('all_images:', all_images)
        print('video_names:', video_names)
        print('frame_nums:', frame_nums)
        print('video_names_uniq:', video_names_uniq)
        print('frame_nums_uniq:', frame_nums_uniq)
        print('vids_and_frames:', vids_and_frames)
        print('frames_distribution:', frames_distribution, end='\n\n')

        return vids_and_frames

    def pad_frames(self, mode):
        frames_dir = None
        if mode == 'train':
            frames_dir = self.train_dir
        elif mode == 'val':
            frames_dir = self.val_dir

        print('Target train:', int(max(self.vids_and_frames.values())))
        print('Target val:', int(max(self.vids_and_frames_val.values())))

        img_black = np.zeros((35, 70, 3))
        print(img_black.shape)

        target_frame_num = max(int(max(self.vids_and_frames.values())), int(max(self.vids_and_frames_val.values())))
        print(target_frame_num)

        for classi in self.classes_dict.keys():
            for vid in self.vids_and_frames.keys():
                if classi == vid.split('_')[0]:
                    if int(self.vids_and_frames[vid]) < target_frame_num:
                        for i in range(1, (target_frame_num - int(self.vids_and_frames[vid]) + 1)):
                            cv2.imwrite(frames_dir + self.classes_dict[classi] + '/' + vid + '_' + 'color' + '_' + str(
                                int(self.vids_and_frames[vid]) + i).zfill(3) + '.jpg', img_black)

# # Val
# all_images_val = []  # 1746
# video_names_val = []  # 1746
# frame_nums_val = []  # 1746
# video_names_uniq_val = []  # 200
# frame_nums_uniq_val = []  # 200
# vids_and_frames_val = {}  # 200
# frames_distribution_val = {}  # 9
# for classi in classes_dict.values():
#     for i in sorted(glob.glob('data/validation/' + classi + '/*.jpg')):
#         all_images_val.append(i)
#
# for img in all_images_val:
#     img_name = img.split('/')[-1].split('.')[0]
#     video_name = img_name[:9]
#     video_names_val.append(video_name)
#     frame_num = img_name[-3:]
#     frame_nums_val.append(frame_num)
#
# video_names_uniq_val = list(sorted(set(video_names_val)))
#
# for i in range(len(video_names_val)):
#     if i < len(video_names_val) - 1:
#         if video_names_val[i] == video_names_val[i + 1]:
#             pass
#         else:
#             frame_nums_uniq_val.append(frame_nums_val[i])
#     else:
#         frame_nums_uniq_val.append(frame_nums_val[-1])
#
# for i in range(len(frame_nums_uniq_val)):
#     vids_and_frames_val[video_names_uniq_val[i]] = frame_nums_uniq_val[i]
#
# for frame_num_uniq_val in sorted(set(frame_nums_uniq_val)):
#     count_d = 0
#     for vid_name_val, frames_num_val in list(vids_and_frames_val.items()):
#         if frames_num_val == frame_num_uniq_val:
#             count_d += 1
#             frames_distribution_val[frame_num_uniq_val] = count_d
#
# print('\nall_images_val:', all_images_val)
# print('video_names_val:', video_names_val)
# print('frame_nums_val:', frame_nums_val)
# print('video_names_uniq_val:', video_names_uniq_val)
# print('frame_nums_uniq_val:', frame_nums_uniq_val)
# print('vids_and_frames_val:', vids_and_frames_val)
# print('frames_distribution_val:', frames_distribution_val)
#
#
# # make sure that padding's shape equals to (w x h x c) of the train images
# img_black = np.zeros((35, 70, 3))
# print(img_black.shape)
#
# target_frame_num = max(int(max(vids_and_frames.values())), int(max(vids_and_frames_val.values())))
# print(target_frame_num)
