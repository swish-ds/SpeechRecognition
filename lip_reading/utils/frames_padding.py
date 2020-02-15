import glob
import os
import cv2
import numpy as np
from utils import mouth_extractor, get_distribution

all_images_val, \
video_names_val, \
frame_nums_val, \
video_names_uniq_val, \
frame_nums_uniq_val, \
vids_and_frames_val, \
frames_distribution_val = get_distribution.get_frames_distribution_val()

train_people, val_people, test_people, classes_num, classes_dict, word_ids = mouth_extractor.split_speakers()

def remove_short_long_train(all_images, video_names, frame_nums, video_names_uniq, frame_nums_uniq, vids_and_frames, frames_distribution):

    comp = {}
    for frame_num_uniq in sorted(set(frame_nums_uniq)):
        comp[frame_num_uniq] = {}
        comp[frame_num_uniq]['class'] = []
        comp[frame_num_uniq]['person'] = []
        comp[frame_num_uniq]['video_names'] = []
        comp[frame_num_uniq]['videos_count'] = 0
        for vid_name, frames_num in list(vids_and_frames.items()):
            if frames_num == frame_num_uniq:
                if vid_name.split('_')[0] not in comp[frame_num_uniq]['class']:
                    comp[frame_num_uniq]['class'].append(vid_name.split('_')[0])
                if vid_name.split('_')[1] not in comp[frame_num_uniq]['person']:
                    comp[frame_num_uniq]['person'].append(vid_name.split('_')[1])
                if vid_name not in comp[frame_num_uniq]['video_names']:
                    comp[frame_num_uniq]['video_names'].append(vid_name)
                    comp[frame_num_uniq]['videos_count'] += 1

    # get the most common number of frames
    maxer = 0
    maxer_key = 0
    for key in frames_distribution:
        if frames_distribution[key] >= maxer:
            maxer = frames_distribution[key]
            maxer_key = key

    #videos either too short or too long
    abnormal_vids = []
    for key in comp:
        for video_name in comp[key]['video_names']:
            if int(key) not in range(int(maxer_key)-2, int(maxer_key)+3):
                abnormal_vids.append(video_name)
    abnormal_vids = sorted(abnormal_vids)

    #remove not needed
    counter = 0
    added = 0
    matched = []
    for classi in list(classes_dict.values()):
        removed = 0
        for abnormal_vid in abnormal_vids:
            for vid in sorted(glob.glob('data/train/' + classi + '/*.jpg')):
                if abnormal_vid == vid.split('/')[-1].split('.')[0].split('_color')[0]:
                    os.remove(vid)
                    #!rm $vid
                    matched.append(abnormal_vid)
                    counter += 1
                    removed += 1
        print(classi, 'Removed', counter, '(' , removed, 'in this dir)')

    print()
    print('Removed', len(set(matched)), 'videos')

    return comp, abnormal_vids


def remove_short_long_val(all_images_val, video_names_val, frame_nums_val, video_names_uniq_val, frame_nums_uniq_val, vids_and_frames_val, frames_distribution_val):
    comp_val = {}
    for frame_num_uniq_val in sorted(set(frame_nums_uniq_val)):
        comp_val[frame_num_uniq_val] = {}
        comp_val[frame_num_uniq_val]['class'] = []
        comp_val[frame_num_uniq_val]['person'] = []
        comp_val[frame_num_uniq_val]['video_names'] = []
        comp_val[frame_num_uniq_val]['videos_count'] = 0
        for vid_name_val, frames_num_val in list(vids_and_frames_val.items()):
            if frames_num_val == frame_num_uniq_val:
                if vid_name_val.split('_')[0] not in comp_val[frame_num_uniq_val]['class']:
                    comp_val[frame_num_uniq_val]['class'].append(vid_name_val.split('_')[0])
                if vid_name_val.split('_')[1] not in comp_val[frame_num_uniq_val]['person']:
                    comp_val[frame_num_uniq_val]['person'].append(vid_name_val.split('_')[1])
                if vid_name_val not in comp_val[frame_num_uniq_val]['video_names']:
                    comp_val[frame_num_uniq_val]['video_names'].append(vid_name_val)
                    comp_val[frame_num_uniq_val]['videos_count'] += 1

    maxer = 0
    maxer_key = 0
    for key in frames_distribution_val:
        if frames_distribution_val[key] >= maxer:
            maxer = frames_distribution_val[key]
            maxer_key = key

    abnormal_vids = []
    for key in comp_val:
        for video_name in comp_val[key]['video_names']:
            if int(key) not in range(int(maxer_key)-1, int(maxer_key)+3):
                abnormal_vids.append(video_name)
    abnormal_vids = sorted(abnormal_vids)

    counter = 0
    added = 0
    matched = []
    for classi in list(classes_dict.values()):
        removed = 0
        for abnormal_vid in abnormal_vids:
            for vid in sorted(glob.glob('data/validation/' + classi + '/*.jpg')):
                if abnormal_vid == vid.split('/')[-1].split('.')[0].split('_color')[0]:
                    os.remove(vid)
                    matched.append(abnormal_vid)
                    counter += 1
                    removed += 1
        print(classi, 'Removed', counter, '(' , removed, 'in this dir)')

    print()
    print('Removed', len(set(matched)), 'videos')

    return comp_val, abnormal_vids


def pad_train(target_frame_num, classes_dict, vids_and_frames):
    img_black = np.zeros((70,140,3))

    for classi in classes_dict.keys():
        for vid in vids_and_frames.keys():
            if classi == vid.split('_')[0]:
                if int(vids_and_frames[vid]) < target_frame_num:
                    for i in range(1, (target_frame_num - int(vids_and_frames[vid]) + 1)):
                        cv2.imwrite('data/train/' + classes_dict[classi] + '/' + vid + '_' + 'color' + '_' + str(int(vids_and_frames[vid]) + i).zfill(3) + '.jpg', img_black)

def pad_val(target_frame_num_val, classes_dict, vids_and_frames_val):
    img_black = np.zeros((70,140,3))
    
    for classi in classes_dict.keys():
        for vid in vids_and_frames_val.keys():
            if classi == vid.split('_')[0]:
                if int(vids_and_frames_val[vid]) < target_frame_num_val:
                    for i in range(1, (target_frame_num_val - int(vids_and_frames_val[vid]) + 1)):
                        cv2.imwrite('data/validation/' + classes_dict[classi] + '/' + vid + '_' + 'color' + '_' + str(int(vids_and_frames_val[vid]) + i).zfill(3) + '.jpg', img_black)

