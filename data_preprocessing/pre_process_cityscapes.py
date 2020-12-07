"""Cityscapes Dataset Preprocessing

Author: Weikun Han <weikunhan@gmail.com>

Reference: 
- https://github.com/fuy34/superpixel_fcn/blob/master/data_preprocessing/pre_process_bsd500.py
"""

import os
import glob
import argparse


def make_dataset(dir):
    cwd = os.getcwd()
    train_list_path = cwd + '/train.txt'
    val_list_path =  cwd + '/val.txt'
    train_list = []
    val_list = []

    try:
        with open(train_list_path, 'r') as tf:
            train_list_0 = tf.readlines()
            for path in train_list_0:
                img_path = os.path.join(dir, 'BSR/BSDS500/data/images/train', path[:-1]+ '.jpg' )
                if not os.path.isfile(img_path):
                    print('The validate images are missing in {}'.format(os.path.dirname(img_path)))
                    print('Please pre-process the BSDS500 as README states and provide the correct dataset path.')
                    exit(1)
                train_list.append(img_path)

        with open (val_list_path, 'r') as vf:
            val_list_0 = vf.readlines()
            for path in val_list_0:
                img_path = os.path.join(dir, 'BSR/BSDS500/data/images/val', path[:-1]+ '.jpg')
                if not os.path.isfile(img_path):
                    print('The validate images are missing in {}'.format(os.path.dirname(img_path)))
                    print('Please pre-process the BSDS500 as README states and provide the correct dataset path.')
                    exit(1)
                val_list.append(img_path)


    except IOError:
        print ('Error No avaliable list ')
        return

    return train_list, val_list

def main():
    datadir = args.dataset
    train_list, val_list = make_dataset(datadir)
    dump_pth = os.path.abspath(args.dump_root)
    print("data will be saved to {}".format(dump_pth))

    # mutil-thread running for speed
    Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, len(train_list),'train', train_samp) for n, train_samp in enumerate(train_list))
    Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, len(train_list),'val', val_samp) for n, val_samp in enumerate(val_list))

    with open(dump_pth + '/train.txt', 'w') as trnf:
        imfiles = glob(os.path.join(dump_pth, 'train', '*_img.jpg'))
        
        for frame in imfiles:
            trnf.write(frame + '\n')

    with open(dump_pth+ '/val.txt', 'w') as trnf:
        imfiles = glob(os.path.join(dump_pth, 'val', '*_img.jpg'))
        
        for frame in imfiles:
            trnf.write(frame + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="", help="where the filtered dataset is stored")
    parser.add_argument("--dump_root", type=str, default="", help="Where to dump the data")
    parser.add_argument("--num_threads", type=int, default=4, help="number of threads to use")
    args = parser.parse_args()

    main()
