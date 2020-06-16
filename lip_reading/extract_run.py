import os
import dlib
import extract

base_dir = 'data'
train_dir = os.path.join(base_dir, 'train/')
val_dir = os.path.join(base_dir, 'validation/')
test_dir = os.path.join(base_dir, 'test/')

classes = 'Begin, Choose, Connection, Navigation, Next, Previous, Start, Stop, Hello, Web'
# classes = 'Begin, Web'
classes = classes.split(', ')
print('Classes:', classes)

predictor_path = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

train_people = 'F01, F02, F04, F05, F06, F07, F08, F09, F11, M01, M02, M04, M08'.split(', ')
# train_people = 'F01, M01, F09'.split(', ')
val_people = 'F10, M07'.split(', ')
print('Train speakers:', train_people)
print('Val speakers:', val_people)

ext = extract.Extractor(base_dir, train_dir, val_dir, test_dir, classes, detector, predictor, 100,
                        train_people, val_people, test_people=None)

ext.remove_dirs()
ext.create_dirs()
ext.extract_crop(mode='train')
ext.extract_crop(mode='val')
