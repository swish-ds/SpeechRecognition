import numpy as np
import random
from scripts import randomize
from utils import global_params

random.seed(global_params.rn_seed)
np.random.seed(global_params.np_random_seed)

train_dir = global_params.train_dir
val_dir = global_params.val_dir
test_dir = global_params.test_dir

classes = global_params.classes

randomizer = randomize.Randomizer(train_dir=train_dir, val_dir=val_dir, classes=classes)
randomizer.shuffle(mode='train')
randomizer.save_to_csv(mode='train')
print()
randomizer.shuffle(mode='val')
randomizer.save_to_csv(mode='val')
