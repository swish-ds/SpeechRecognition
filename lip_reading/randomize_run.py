import os
import glob
import pandas as pd
from tqdm import tqdm

# randomize train data
base_dir = 'data'
train_dir = os.path.join(base_dir, 'train/')
val_dir = os.path.join(base_dir, 'validation/')
test_dir = os.path.join(base_dir, 'test/')

classes = 'Begin, Choose, Connection, Navigation, Next, Previous, Start, Stop, Hello, Web'
# classes = 'Begin, Web'
classes = classes.split(', ')
classes_num = ['0' + str(i) if i < 10 else str(i) for i in range(1, len(classes)+1)]
classes_dict = dict(zip(classes_num, classes))

for classi in classes:
    matched = 0
    video_names = []
    frames = glob.glob('data/train/' + classi + '/' + '*.jpg')

    for frame in frames:
        img_name = frame.split('/')[-1].split('.')[0]
        video_name = img_name.split('_color')[0]
        video_names.append(video_name)

    video_names_uniq = list(set(video_names))

    for i in range(len(video_names_uniq)):
        video_names_uniq[i] = str(i).zfill(6) + '_' + video_names_uniq[i]

    frames = sorted(glob.glob('data/train/' + classi + '/' + '*.jpg'))

    for i in range(len(video_names_uniq)):
        for frame in frames:
            if video_names_uniq[i].split(str(i).zfill(6) + '_')[1] == frame.split('/')[-1].split('.')[0].split('_color')[0]:
                os.rename(frame, frame.split('/')[0] + '/' + frame.split('/')[1] + '/' + frame.split('/')[2] + '/' + str(i).zfill(6) + '_' + frame.split('/')[-1])

train_image = []
train_class = []

for class_id in tqdm(range(len(classes))):
    images = sorted(glob.glob('data/train/' + classes[class_id] + '/*.jpg'))
    for i in range(len(images)):
        if ('noised' in images[i] or 'rand_contr' in images[i]
                or 'vert_flip' in images[i] or 'hor_flip' in images[i]):
            # print(images[i].split('/')[3])
            # print(classes_dict[images[i].split('/')[3].split(']')[-1][:2]])
            train_image.append(images[i].split('/')[3])
            train_class.append(classes_dict[images[i].split('/')[3].split(']')[-1][:2]])
        else:
            # print(images[i].split('/')[3])
            # print(images[i].split('/')[3].split('_')[1])
            # print(classes_dict[images[i].split('/')[3].split('_')[1]])
            train_image.append(images[i].split('/')[3])
            train_class.append(classes_dict[images[i].split('/')[3].split('_')[1]])

train_data = pd.DataFrame()
train_data['image'] = train_image
train_data['class'] = train_class

train_data.to_csv('data/miracl/train_new.csv', header=True, index=False)

train = pd.read_csv('data/miracl/train_new.csv')
print(train.head())
print((train.tail()))