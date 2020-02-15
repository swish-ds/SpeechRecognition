import cv2
import glob
import os
import numpy as np
from albumentations.augmentations import transforms

def augment_frames(classes):

    def get_idx(classi):
        start_idx = (len('data/train/') + len(classi)+1)
        return start_idx


    for classi in classes:
        files = [file for file in sorted(glob.glob(os.path.join('data/train/' + classi + '/*.jpg')))]
        files_clean = [file_clean for file_clean in sorted(files)
                    if 'vert_flip' not in file_clean
                    and 'hor_flip' not in file_clean
                    and 'rot_45' not in file_clean
                    and 'rot_315' not in file_clean
                    and 'rot_135' not in file_clean
                    and 'rot_225' not in file_clean
                    and 'rot_90' not in file_clean
                    and 'rot_270' not in file_clean
                    and 'rand_contr' not in file_clean
                    and 'noised' not in file_clean]
        files_rem = [file_rem for file_rem in sorted(files)
                    if 'vert_flip' in file_rem
                    or 'hor_flip' in file_rem
                    or 'rot_45' in file_rem
                    or 'rot_315' in file_rem
                    or 'rot_135' in file_rem
                    or 'rot_225' in file_rem
                    or 'rot_90' in file_rem
                    or 'rot_270' in file_rem
                    or 'rand_contr' in file_rem
                    or 'noised' in file_rem]
        for file_to_rem in files_rem: os.remove(file_to_rem)
        files = [file for file in sorted(glob.glob(os.path.join('data/train/' + classi + '/*.jpg')))]
        print(classi, 'dir: removed', len(files_rem), 'files.', len(files), 'to be augmented')

        for file_to_hor_flip in files_clean:
            img = cv2.imread(file_to_hor_flip, 1)
            img = transforms.HorizontalFlip().apply(img)
            file_to_write = file_to_hor_flip[:get_idx(classi)] + '[hor_flip]' + file_to_hor_flip[get_idx(classi):]
            cv2.imwrite(file_to_write, img)
        files = [file for file in sorted(glob.glob(os.path.join('data/train/' + classi + '/*.jpg')))]
        files_hor_flipped = [file_hor_flipped for file_hor_flipped in sorted(files) if 'hor_flip' in file_hor_flipped]
        print('Horizontally flipped', len(files_hor_flipped), 'files.', len(files), 'in the directory')
        
        for file_to_vert_flip in files_clean:
            img = cv2.imread(file_to_vert_flip, 1)
            img = transforms.VerticalFlip().apply(img)
            file_to_write = file_to_vert_flip[:get_idx(classi)] + '[vert_flip]' + file_to_vert_flip[get_idx(classi):]
            cv2.imwrite(file_to_write, img)
        files = [file for file in sorted(glob.glob(os.path.join('data/train/' + classi + '/*.jpg')))]
        files_vert_flipped = [file_vert_flipped for file_vert_flipped in sorted(files) if 'vert_flip' in file_vert_flipped]
        print('Vertically flipped', len(files_vert_flipped), 'files.', len(files), 'in the directory')
        
        #files_not_noised = [f for f in sorted(glob.glob(os.path.join('data/train/' + classi + '/*.jpg'))) if 'noised' not in f ]    
        #for file_to_rand_contr in files_not_noised:
        #    img = cv2.imread(file_to_rand_contr, 1)
        #    img = transforms.RandomContrast().apply(img)
        #    file_to_write = file_to_rand_contr[:get_idx(classi)] + '[rand_contr]' + file_to_rand_contr[get_idx(classi):]
        #    #print(file_to_rand_contr)
        #    #print(file_to_write)
        #    cv2.imwrite(file_to_write, img)
        #files = [file for file in sorted(glob.glob(os.path.join('data/train/' + classi + '/*.jpg')))]
        #files_rand_contred = [file_rand_contred for file_rand_contred in sorted(files) if 'rand_contr' in file_rand_contred]
        #print('Random contrasted', len(files_rand_contred), 'files.', len(files), 'in the directory')
        
        files_not_rand_contred = [file_not_rand_contred for file_not_rand_contred in sorted(files) if 'rand_contr' not in file_not_rand_contred]
        for file_to_noise in files_not_rand_contred:
            if '[hor_flip]' in file_to_noise:
                img = cv2.imread(file_to_noise, 1)
                gauss = np.random.uniform(0, 64, img.size)
                gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
                img = cv2.add(img,gauss)
                file_to_write = file_to_noise[:get_idx(classi)] + '[noised]' + file_to_noise[get_idx(classi):]
                cv2.imwrite(file_to_write, img)
            elif '[vert_flip]' in file_to_noise:
                img = cv2.imread(file_to_noise, 1)
                gauss = np.random.uniform(0, 64, img.size)
                gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
                img = cv2.add(img,gauss)
                file_to_write = file_to_noise[:get_idx(classi)] + '[noised]' + file_to_noise[get_idx(classi):]
                cv2.imwrite(file_to_write, img)
            else:
                img = cv2.imread(file_to_noise, 1)
                gauss = np.random.uniform(0, 64, img.size)
                gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
                img = cv2.add(img,gauss)
                file_to_write = file_to_noise[:get_idx(classi)] + '[noised]' + file_to_noise[get_idx(classi):]
                cv2.imwrite(file_to_write, img)
        files = [file for file in sorted(glob.glob(os.path.join('data/train/' + classi + '/*.jpg')))]
        files_noised = [file_noised for file_noised in sorted(files) if 'noised' in file_noised]
        print('Noised', len(files_noised), 'files.', len(files), 'in the directory')
        
        print()