import dlib
import os

# for random seeds (train_run, randomize_run)
rn_seed = 0
np_random_seed = 0
tf_random = 0

# for extract_run, padding_run, randomize_run
repo_dir = os.environ['LRDIR']
base_dir = os.path.join(os.environ['LRDIR'][:-1], 'lip_reading/data/')
train_dir = os.path.join(base_dir[:-1], 'train/')
val_dir = os.path.join(base_dir[:-1], 'validation/')
test_dir = os.path.join(base_dir[:-1], 'test/')
classes = 'Begin, Choose, Connection, Navigation, Next, Previous, Start, Stop, Hello, Web'
# classes = 'Автомобиль, Аудитория, Взаимодействие, Вклад, Океан, Офис, Привет, Процент, Ручка, Ставка'
classes = classes.split(', ')

# for extract_run
predictor_path = os.path.join(os.environ['LRDIR'][:-1], 'lip_reading/utils/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
scale = 100
train_people = 'F01, F02, F04, F05, F06, F07, F08, F09, F11, M01, M02, M04, M08'.split(', ')
val_people = 'F10, M07'.split(', ')
# train_people = 'F01, F02, F03, F04, F05, F06, F07, F08, F09, F10, F12, F13, M01'.split(', ')
# val_people = 'F14, F15'.split(', ')
size_x = 140
size_y = 70

# for train_run
model_type = 'norm'     # can change
optimizer = 'ada'
epochs = 300
lr = 1e-4
mom = 0.9
batch_s = 10
classes_n = 10
dropout_s = 0.5
frames_n = 25
img_w = 140
img_h = 70
img_c = 3
