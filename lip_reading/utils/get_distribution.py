import glob
import numpy as np
import matplotlib.pyplot as plt
from utils import mouth_extractor

train_people, val_people, test_people, classes_num, classes_dict, word_ids = mouth_extractor.split_speakers()

def get_frames_distribution_train():
    
    all_images = []
    video_names = []
    frame_nums = []
    video_names_uniq = []
    frame_nums_uniq = []
    vids_and_frames = {}
    frames_distribution = {}

    for classi in classes_dict.values():
        for i in sorted(glob.glob('data/train/' + classi + '/*.jpg')):
            all_images.append(i)
            
    for img in all_images:
        img_name = img.split('/')[-1].split('.')[0]
        video_name = img_name[:9]
        video_names.append(video_name)
        frame_num = img_name[-3:]
        frame_nums.append(frame_num)
        
    video_names_uniq = list(sorted(set(video_names)))

    for i in range(len(video_names)):
        if i < len(video_names)-1:
            if video_names[i] == video_names[i+1]:
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

    return all_images, video_names, frame_nums, video_names_uniq, frame_nums_uniq, vids_and_frames, frames_distribution


def get_frames_distribution_val():
    
    all_images_val = []
    video_names_val = []
    frame_nums_val = []
    video_names_uniq_val = []
    frame_nums_uniq_val = []
    vids_and_frames_val = {}
    frames_distribution_val = {}

    for classi in classes_dict.values():
        for i in sorted(glob.glob('data/validation/' + classi + '/*.jpg')):
            all_images_val.append(i)
            
    for img in all_images_val:
        img_name = img.split('/')[-1].split('.')[0]
        video_name = img_name[:9]
        video_names_val.append(video_name)
        frame_num = img_name[-3:]
        frame_nums_val.append(frame_num)
        
    video_names_uniq_val = list(sorted(set(video_names_val)))

    for i in range(len(video_names_val)):
        if i < len(video_names_val)-1:
            if video_names_val[i] == video_names_val[i+1]:
                pass
            else:
                frame_nums_uniq_val.append(frame_nums_val[i])
        else:
            frame_nums_uniq_val.append(frame_nums_val[-1])
            
    for i in range(len(frame_nums_uniq_val)):
            vids_and_frames_val[video_names_uniq_val[i]] = frame_nums_uniq_val[i]
            
    for frame_num_uniq_val in sorted(set(frame_nums_uniq_val)):
        count_d = 0
        for vid_name_val, frames_num_val in list(vids_and_frames_val.items()):
            if frames_num_val == frame_num_uniq_val:
                count_d += 1
                frames_distribution_val[frame_num_uniq_val] = count_d

    return all_images_val, video_names_val, frame_nums_val, video_names_uniq_val, frame_nums_uniq_val, vids_and_frames_val, frames_distribution_val


def plot_distribution(frames_distribution, frames_distribution_val, interval = 1):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,8))

    ax1.bar(list(frames_distribution.keys()), [int(f) for f in frames_distribution.values()], 0.7)
    ax1.grid(axis = 'y')
    ax1.set_yticks(np.arange(0, max(frames_distribution.values())+2, interval))

    ax2.bar(list(frames_distribution_val.keys()), [int(f) for f in frames_distribution_val.values()], 0.7)
    ax2.grid(axis = 'y')
    ax2.set_yticks(np.arange(0, max(frames_distribution_val.values())+2, interval)) 

    fig.show()