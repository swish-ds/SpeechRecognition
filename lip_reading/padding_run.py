import padding
import os

base_dir = 'data'
train_dir = os.path.join(base_dir, 'train/')
val_dir = os.path.join(base_dir, 'validation/')
test_dir = os.path.join(base_dir, 'test/')

classes = 'Begin, Choose, Connection, Navigation, Next, Previous, Start, Stop, Hello, Web'
# classes = 'Begin, Web'
classes = classes.split(', ')
classes_num = ['0' + str(i) if i < 10 else str(i) for i in range(1, len(classes)+1)]
classes_dict = dict(zip(classes_num, classes))

padder = padding.Pad(train_dir, val_dir, classes)
padder.pad_frames(mode='train')
padder.pad_frames(mode='val')
