import dlib
import os

# for random seeds (train_run, randomize_run)
rn_seed = 0
np_random_seed = 0
tf_random = 0

# for extract_run, padding_run, randomize_run
repo_dir = os.environ['LRDIR']
base_dir = os.path.join(os.environ['LRDIR'], 'lip_reading/data/')
train_dir = os.path.join(base_dir, 'train/')
val_dir = os.path.join(base_dir, 'validation/')
test_dir = os.path.join(base_dir, 'test/')
classes = 'Begin, Choose, Connection, Navigation, Next, Previous, Start, Stop, Hello, Web'
classes = classes.split(', ')

# for extract_run
predictor_path = os.path.join(os.environ['LRDIR'][:-1], 'lip_reading/utils/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
scale = 200
train_people = 'F01, F02, F04, F05, F06, F07, F08, F09, F11, M01, M02, M04, M08'.split(', ')
val_people = 'F10, M07'.split(', ')
size_x = 70
size_y = 35

# for train_run
model_type = 'norm'     # can change
optimizer = 'sgd'       # can change
epochs = 23            # can change+
lr = 4e-3
mom = 0.9
batch_s = 10            # can change+
classes_n = 10          # can change+
dropout_s = 0.5         # can change
frames_n = 22           # can change+
img_w = int(size_x * scale / 100)
img_h = int(size_y * scale / 100)
img_c = 3
