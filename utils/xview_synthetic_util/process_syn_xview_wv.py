'''
xview process
in order to generate xivew 2-d background with synthetic airplances
'''
import glob
import numpy as np
import argparse
import os
from PIL import Image
import pandas as pd
import shutil


def get_black_raw_img_list():
    '''
    get raw images contain more than 30% black pixels
    '''
    tif_files = np.sort(glob.glob(os.path.join(args.image_folder, '*.tif')))
    f_txt = open(os.path.join(args.raw_folder, 'black_tif_names.txt'), 'w')
    black_thr = 0.3
    for tf in tif_files:
        im = Image.open(tf)
        im_gray = np.array(im.convert('L'))
        non_black_percent = np.count_nonzero(im_gray) / (im_gray.shape[0] * im_gray.shape[1])

        # NOTE: when the image is full of black pixel  or larger than (1-thr) covered with black pixel
        if non_black_percent < black_thr:
            f_txt.write('%s\n' % tf)
    f_txt.close()


def get_tif_contain_airplanes():
    f_txt = open(os.path.join(args.raw_folder, 'airplane_tif_names.txt'), 'w')

    df_files = pd.read_csv('/media/lab/Yang/data/xView_YOLO/labels/608/1_cls/image_names_608_1cls.csv')
    file_names = [x.split('_')[0] + '.tif' for x in df_files['file_name']]
    file_names = list(set(file_names))
    file_names.sort()
    for fn in file_names:
        f_txt.write('%s\n' % fn)
    f_txt.close()
    print(len(file_names))


def get_no_airplane_raw_images():
    df_files = pd.read_csv(os.path.join(args.raw_folder, 'airplane_tif_names.txt'), header=None).to_numpy()
    airplane_files = df_files[:, 0]
    save_dir = os.path.join(args.raw_folder, 'no_airplanes')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    tif_files = np.sort(glob.glob(os.path.join(args.image_folder, '*.tif')))
    for f in tif_files:
        name = os.path.basename(f)

        if name not in airplane_files:
            shutil.copy(f, os.path.join(save_dir, name))


def get_shape_raw_images():
    '''
    get raw images contain more than 30% black pixels
    '''
    tif_files = np.sort(glob.glob(os.path.join(args.raw_folder, 'no_airplanes', '*.tif')))
    f_txt = open(os.path.join(args.raw_folder, 'no_ariplanes_raw_images_size.txt'), 'w')
    for tf in tif_files:
        im = Image.open(tf)
        im_gray = np.array(im.convert('L'))

        f_txt.write('%d %d\n' % (im_gray.shape[0], im_gray.shape[1]))
    f_txt.close()
    arr_sizes = np.loadtxt(os.path.join(args.raw_folder, 'no_ariplanes_raw_images_size.txt'))
    print(np.min(arr_sizes, axis=0))
    print(np.max(arr_sizes, axis=0))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str,
                        help="Path to folder containing image chips (ie 'Image_Chips/') ",
                        default='/media/lab/Yang/data/xView/train_images/')
    parser.add_argument("--raw_folder", type=str,
                        help="Path to folder containing raw images ",
                        default='/media/lab/Yang/data/xView/')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    '''
    get raw images contain more than 30% black pixels
    '''
    # get_black_raw_img_list()

    # get_tif_contain_airplanes()

    # get_no_airplane_raw_images()

    # get_shape_raw_images()






