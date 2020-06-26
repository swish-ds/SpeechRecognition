from scripts import padding
from utils import global_params

train_dir = global_params.train_dir
val_dir = global_params.val_dir
test_dir = global_params.test_dir

classes = global_params.classes

padder = padding.Pad(train_dir=train_dir, val_dir=val_dir, classes=classes)
padder.pad_frames(mode='train')
padder.pad_frames(mode='val')
