import dlib
import cv2
import glob
import os
import cv2
import shutil
from tqdm import tqdm

classes = 'Begin, Choose, Connection, Navigation, Next, Previous, Start, Stop, Hello, Web'
classes = classes.split(', ')

def create_dirs():
    base_dir = 'data'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    if os.path.isdir(train_dir) or os.path.isdir(val_dir) or os.path.isdir(test_dir):
        shutil.rmtree(train_dir)
        shutil.rmtree(val_dir)
        shutil.rmtree(test_dir)
        os.makedirs(train_dir)
        os.makedirs(val_dir)
        os.makedirs(test_dir)
    else:
        os.makedirs(train_dir)
        os.makedirs(val_dir)
        os.makedirs(test_dir)
    
    for class_name in classes:
        train_vids_dir = os.path.join(train_dir, class_name)
        val_vids_dir = os.path.join(val_dir, class_name)
        test_vids_dir = os.path.join(test_dir, class_name)
        
        os.makedirs(train_vids_dir)
        os.makedirs(val_vids_dir)
        os.makedirs(test_vids_dir)
    
def split_speakers():
    train_people = 'F01, F02, F04, F05, F06, F07, F08, F09, M01, M04'.split(', ')
    val_people = 'F10, M07'.split(', ')
    test_people = 'F11, M08'.split(', ')

    classes_num = ['0'+str(i) if i < 10 else str(i) for i in range(1, 11) ]
    classes_dict = dict(zip(classes_num, classes))
    word_ids = ['0'+str(i) if i < 10 else str(i) for i in range(1, 11) ]

    return train_people, val_people, test_people, classes_num, classes_dict, word_ids

def define_predictor_detector(predictor_location):
    predictor = dlib.shape_predictor(predictor_location)
    detector = dlib.get_frontal_face_detector()

    return predictor, detector

train_people, val_people, test_people, classes_num, classes_dict, word_ids = split_speakers()
predictor, detector = define_predictor_detector('assests/predictors/shape_predictor_68_face_landmarks.dat')

def extract_train(classes_n = len(classes_num), people_n = len(train_people), words_n = len(word_ids)):
    counter = 0
    for classi in tqdm(classes_num[:classes_n]):
        for person in train_people[:people_n]:
            for word_id in word_ids[:words_n]:
                for f in sorted(glob.glob(os.path.join('data/miracl/'+person+'/words/'+classi+'/'+ word_id, "*.jpg"))):
                    img = cv2.imread(f, 1)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    rects = detector(gray)
                    
                    for k, rect in enumerate(rects):
                        shape = predictor(gray, rect)
                        
                        x_51 = shape.part(51).x
                        y_51 = shape.part(51).y
                        x_57 = shape.part(57).x
                        y_57 = shape.part(57).y

                        x1_m = x_51 - 18
                        y1_m = y_51 - 7
                        x2_m = x_57 + 18
                        y2_m = y_57 + 9

                        offset_x_m = (70-(abs(x1_m-x2_m)))/2
                        offset_y_m = (35-(abs(y1_m-y2_m)))/2

                        img = img[int(y1_m-offset_y_m):int(y2_m+offset_y_m), int(x1_m-offset_x_m):int(x2_m+offset_x_m)]
                        
                        scale_percent = 200
                        width = int(img.shape[1] * scale_percent / 100)
                        height = int(img.shape[0] * scale_percent / 100)
                        dim = (width, height)
                        img = cv2.resize(img, (int(img.shape[1]*200/100), int(img.shape[0]*200/100)), interpolation=cv2.INTER_AREA) 
                        
                    counter += 1
                    
                    cv2.imwrite('data/train/' + classes_dict[classi] + '/' + classi  + '_' + person  + '_' + word_id + '_' + f[28:-4] + '.jpg', img)

def extract_val(classes = len(classes_num), people = len(val_people), words = len(word_ids)):
    counter = 0
    for classi in tqdm(classes_num[:classes]):
        for person in val_people[:people]:
            for word_id in word_ids[:words]:
                for f in sorted(glob.glob(os.path.join('data/miracl/'+person+'/words/'+classi+'/'+ word_id, "*.jpg"))):
                    img = cv2.imread(f, 1)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    rects = detector(gray)
                    
                    for k, rect in enumerate(rects):
                        shape = predictor(gray, rect)
                        
                        x_51 = shape.part(51).x
                        y_51 = shape.part(51).y
                        x_57 = shape.part(57).x
                        y_57 = shape.part(57).y

                        x1_m = x_51 - 18
                        y1_m = y_51 - 7
                        x2_m = x_57 + 18
                        y2_m = y_57 + 9

                        offset_x_m = (70-(abs(x1_m-x2_m)))/2
                        offset_y_m = (35-(abs(y1_m-y2_m)))/2

                        img = img[int(y1_m-offset_y_m):int(y2_m+offset_y_m), int(x1_m-offset_x_m):int(x2_m+offset_x_m)]
                        
                        scale_percent = 200
                        width = int(img.shape[1] * scale_percent / 100)
                        height = int(img.shape[0] * scale_percent / 100)
                        dim = (width, height)
                        img = cv2.resize(img, (int(img.shape[1]*200/100), int(img.shape[0]*200/100)), interpolation=cv2.INTER_AREA) 
                        
                    counter += 1
                    
                    cv2.imwrite('data/validation/' + classes_dict[classi] + '/' + classi  + '_' + person  + '_' + word_id + '_' + f[28:-4] + '.jpg', img)