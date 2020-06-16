import os

base_dir = 'data'
train_dir = os.path.join(base_dir, 'train/')
val_dir = os.path.join(base_dir, 'validation/')
test_dir = os.path.join(base_dir, 'test/')

print(os.path.isdir(train_dir))
