from utils import global_params
from utils.keras_video_datagen import ImageDataGenerator
import os
import cv2
import matplotlib.pyplot as plt

datagen = ImageDataGenerator()
train_data = datagen.flow_from_directory(os.path.join(global_params.repo_dir, "lip_reading/data/train_mir"),
                                         augm=False,
                                         target_size=(70, 140),
                                         batch_size=1,
                                         frames_per_step=22, shuffle=True, seed=0,

                                        color_mode='rgb')
cnt = 0
for i in range(15):
    x, y, _ = train_data.next()

    fig=plt.figure(figsize=(15, 15))
    columns = 4
    rows = 7
    for i in range(1, len(x[0])+1):
        img = x[0][i-1]
        # print(img)
        # img += np.random.normal(0, 1.0 * img.std(), img.shape)
        # img = np.clip(img, 0, 255)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img.astype('uint8'))
        # plt.imsave(os.path.join(global_params.repo_dir, "lip_reading/pictures/augs/img_%s.png" % str(cnt)), img.astype('uint8'))
    print(cnt)
    plt.savefig(os.path.join(global_params.repo_dir, "lip_reading/pictures/augs_mir/aug_mir_%s.png" % str(cnt)))
    # plt.show()
    cnt += 1



