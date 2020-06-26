from scripts import extract
from utils import global_params

train_dir = global_params.train_dir
val_dir = global_params.val_dir
test_dir = global_params.test_dir

classes = global_params.classes
print('Classes:', classes)

predictor_path = global_params.predictor_path
detector = global_params.detector
predictor = global_params.predictor

train_people = global_params.train_people
val_people = global_params.val_people
print('Train speakers:', train_people)
print('Val speakers:', val_people)

size_x = global_params.size_x
size_y = global_params.size_y
scale = global_params.scale

ext = extract.Extractor(train_dir=train_dir, val_dir=val_dir, test_dir=test_dir, classes=classes, detector=detector,
                        predictor=predictor, scale=scale,
                        train_people=train_people, val_people=val_people, size_x=size_x, size_y=size_y,
                        test_people=None)

ext.remove_dirs()
ext.create_dirs()
ext.extract_crop(mode='train')
ext.extract_crop(mode='val')
