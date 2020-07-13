from utils import global_params
import tqdm
import glob
import os
import cv2
import statistics
import moviepy
from moviepy.editor import VideoFileClip
from moviepy.video.fx import speedx

classes = global_params.classes
classes_num = ['0' + str(i) if i < 10 else str(i) for i in range(1, len(classes) + 1)]
word_ids = ['0' + str(i) if i < 10 else str(i) for i in range(1, 11)]
classes_dict = dict(zip(classes_num, classes))
people_list = global_params.train_people + global_params.val_people


def equalize_length():
    for classi in classes_num[:]:
        print(classi, classes_dict[classi], end='\n\n')

        for person in people_list[:]:
            for word_id in word_ids[:]:
                for video_file in sorted(
                        glob.glob(
                            os.path.join('data/miracl/' + person + '/words/' + classi + '/' + word_id, 'color.mp4'))):
                    count = 1

                    print(video_file)

                    clip = VideoFileClip(video_file)
                    modfile = video_file.split('.')[0] + '_mod.mp4'
                    clip = speedx.speedx(clip, final_duration=2.0)
                    clip.write_videofile(modfile, fps=25, logger=None)

                    cap = cv2.VideoCapture(modfile)
                    frame_rate = cap.get(cv2.CAP_PROP_FPS)

                    while cap.isOpened():
                        frame_id = cap.get(1)
                        success, frame = cap.read()
                        if not success:
                            break
                        if not frame_id % 2:
                            filename = 'data/miracl/' + person + '/words/' + classi + '/' + word_id + '/color_%s.jpg' % str(count).zfill(3)
                            cv2.imwrite(filename, frame)
                            count += 1

                    cap.release()
                    # os.remove(modfile)
        print('\n\n\n')

equalize_length()

def del_files():
    for classi in classes_num[:]:
        print(classi, classes_dict[classi], end='\n\n')

        for person in people_list[:]:
            for word_id in word_ids[:]:
                for video_file in sorted(
                        glob.glob(
                            os.path.join('data/miracl/' + person + '/words/' + classi + '/' + word_id, '*'))):
                    if 'jpg' in video_file or 'mod' in video_file:
                        print(video_file)
                        os.remove(video_file)
        print('\n\n\n')

# del_files()



