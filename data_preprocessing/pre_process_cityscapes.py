"""Cityscapes Dataset Preprocessing

Author: Weikun Han <weikunhan@gmail.com>

Reference: 
- https://github.com/fuy34/superpixel_fcn/blob/master/data_preprocessing/pre_process_bsd500.py
"""

import cv2
import os
import glob
import shutil
import argparse
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm

def make_dataset(data_dir, image_flag=True):
    train_list = []
    val_list = []
    train_folder_list = glob.glob(os.path.join(data_dir, 'train', '*'))
    val_folder_list = glob.glob(os.path.join(data_dir, 'val', '*'))
    
    if image_flag: 
        for folder_path in train_folder_list:
            for file_path in glob.glob(os.path.join(folder_path, '*.png')):
                train_list.append(file_path)

        for folder_path in val_folder_list:
            for file_path in glob.glob(os.path.join(folder_path, '*.png')):
                val_list.append(file_path)    
    else:
        for folder_path in train_folder_list:
            for file_path in glob.glob(os.path.join(folder_path, '*_labelIds.png')):
                train_list.append(file_path)       

        for folder_path in val_folder_list:
            for file_path in glob.glob(os.path.join(folder_path, '*_labelIds.png')):
                val_list.append(file_path)       

    if len(train_list) != 2975:
        raise AssertionError('The training dataset is supported to have 2975 samples!')

    if len(val_list) != 500:
        raise AssertionError('The training dataset is supported to have 500 samples!')
    
    return train_list, val_list

def save_images(image_patch_in, image_path_out):
    if os.path.isfile(image_path_out):
        try:
            Image.open(image_path_out).verify()

            return 
        except Exception as e:
            pass

    image = cv2.imread(image_patch_in)
    cv2.imwrite(image_path_out, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def generate_image_dataset(train_data_dir_out, val_data_dir_out,
                           image_train_list, image_val_list):
    pool = Pool() 

    for file_path in tqdm(image_train_list, total= len(image_train_list), 
                          desc='Start train image proccessing'):
        folder_name = file_path.split(os.path.sep)[-2]
        temp_folder_path = os.path.join(train_data_dir_out, folder_name)
        file_name = file_path.split(os.path.sep)[-1].split('.')[0][:-12] + '_img.jpg'
        temp_file_path = os.path.join(temp_folder_path, file_name)

        if not os.path.exists(temp_folder_path):
            os.mkdir(temp_folder_path)
        
        pool.apply_async(save_images, (file_path, temp_file_path))

    for file_path in tqdm(image_val_list, total= len(image_val_list), 
                          desc='Start val image proccessing'):
        folder_name = file_path.split(os.path.sep)[-2]
        temp_folder_path = os.path.join(val_data_dir_out, folder_name)
        file_name = file_path.split(os.path.sep)[-1].split('.')[0][:-12] + '_img.jpg'
        temp_file_path = os.path.join(temp_folder_path, file_name)

        if not os.path.exists(temp_folder_path):
            os.mkdir(temp_folder_path)
        
        pool.apply_async(save_images, (file_path, temp_file_path))
        
    pool.close()
    pool.join()

def generate_label_dataset(train_data_dir_out, val_data_dir_out,
                           lable_train_list, label_val_list):
    for file_path in tqdm(lable_train_list, total= len(lable_train_list), 
                           desc='Start train label proccessing'):
        folder_name = file_path.split(os.path.sep)[-2]
        file_name = file_path.split(os.path.sep)[-1].split('.')[0][:-16] + '_label.png'
        temp_file_path = os.path.join(train_data_dir_out, folder_name, file_name)
        shutil.copyfile(file_path, temp_file_path)

    for file_path in tqdm(label_val_list, total= len(label_val_list), 
                          desc='Start val label proccessing'):
        folder_name = file_path.split(os.path.sep)[-2]
        file_name = file_path.split(os.path.sep)[-1].split('.')[0][:-16] + '_label.png'
        temp_file_path = os.path.join(val_data_dir_out, folder_name, file_name)
        shutil.copyfile(file_path, temp_file_path)
    
def main():
    data_dir_in = args.dataset
    data_dir_out = os.path.abspath(args.dump_root)

    print("Data will be saved to: {}".format(data_dir_out))

    image_data_dir_in = os.path.join(data_dir_in, 'leftImg8bit')
    label_data_dir_in = os.path.join(data_dir_in, 'gtFine')
    train_data_dir_out = os.path.join(data_dir_out, 'train')
    val_data_dir_out = os.path.join(data_dir_out, 'val')

    if not os.path.exists(train_data_dir_out):
        os.mkdir(train_data_dir_out)

    if not os.path.exists(val_data_dir_out):
        os.mkdir(val_data_dir_out)

    image_train_list, image_val_list = make_dataset(image_data_dir_in, image_flag=True)
    lable_train_list, label_val_list = make_dataset(label_data_dir_in, image_flag=False)
    generate_image_dataset(train_data_dir_out, val_data_dir_out,
                           image_train_list, image_val_list)
    generate_label_dataset(train_data_dir_out, val_data_dir_out,
                           lable_train_list, label_val_list)

    with open(os.path.join(data_dir_out, 'train.txt'), 'w') as f:
        for folder_path in glob.glob(os.path.join(train_data_dir_out, '*')):
            for file_path in glob.glob(os.path.join(folder_path, '*_img.jpg')):
                f.write(file_path + '\n')

    with open(os.path.join(data_dir_out, 'val.txt'), 'w') as f:
        for folder_path in glob.glob(os.path.join(val_data_dir_out, '*')):
            for file_path in glob.glob(os.path.join(folder_path, '*_img.jpg')):
                f.write(file_path + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="", help="where the filtered dataset is stored")
    parser.add_argument("--dump_root", type=str, default="", help="Where to dump the data")
    args = parser.parse_args()

    main()
