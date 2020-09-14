import json
import numpy as np
from matplotlib import pyplot as plt
import os
from glob import glob
from skimage.color import rgb2gray
from skimage import io
import shutil
import pandas as pd
import sys
sys.path.append('/data/users/yang/code/yxu-yolov3-xview/')
import argparse
from utils.xview_synthetic_util import preprocess_xview_syn_data_distribution as pps

def compare_overlap(file1, file2, save_file):

    img_file1 = io.imread(file1)
    img_file2 =io.imread(file2)

    res = 0.3
    size = img_file1.shape[0]
    step = size//2

    common1 = img_file1[:, step:]
    common2 = img_file2[:, :step]

    plt.figure(figsize=(15, 8))
    plt.subplot(131)
    plt.imshow(common1)

    plt.subplot(132)
    plt.imshow(common2)

    plt.subplot(133)
    plt.imshow(common2-common1)

    plt.tight_layout()
    plt.show()

    # plt.savefig('/media/lab/Yang/data/synthetic_data/Airplanes/59_60common_change.jpg')

    plt.savefig(save_file)


def compare_images(rgb_dir1, xlbl1, rgb_dir2, xlbl2):
    step = syn_args.tile_size * syn_args.resolution

    save_dir = os.path.join(syn_args.cat_sample_dir, 'compare_rgb_same')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    rgb_dir1 = os.path.join(rgb_dir1, 'color_all_images_step{}'.format(step))
    rgb_dir2 = os.path.join(rgb_dir2, 'color_all_images_step{}'.format(step))

    tx_image_files = np.sort(glob(os.path.join(rgb_dir1, '*.png')))
    tx_img_names = [os.path.basename(f) for f in tx_image_files]

    for ix in range(len(tx_image_files)):
    # for ix in range(10):
        file1 = tx_image_files[ix]
        file2 = os.path.join(rgb_dir2, tx_img_names[ix])

        img_file1 = io.imread(file1)
        img_file2 = io.imread(file2)

        fig, axs = plt.subplots(1, 3, figsize=(15, 8), sharex=True, sharey=True)

        axs[0].imshow(img_file1)
        axs[0].set_title(tx_img_names[ix])
        axs[0].set_xlabel(xlbl1)
        axs[1].imshow(img_file2)
        axs[1].set_title(tx_img_names[ix])
        axs[1].set_xlabel(xlbl2)
        axs[2].imshow(img_file1 - img_file2)
        axs[2].set_xlabel('left - right')
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, tx_img_names[ix]))
        # plt.show()
        # plt.show()
        # exit(0)

def get_part_syn_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_filepath", type=str, help="Filepath to GEOJSON coordinate file",
                        default='/media/lab/Yang/data/xView/xView_train.geojson')

    parser.add_argument("--syn_images_save_dir", type=str, help="rgb images of synthetic airplanes",
                        default='/media/lab/Yang/data/synthetic_data/{}/')
    parser.add_argument("--syn_annos_save_dir", type=str, help="gt of synthetic airplanes",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_{}_cls_xcycwh/')
    parser.add_argument("--syn_txt_save_dir", type=str, help="gt related files of synthetic airplanes",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_{}_cls/')

    parser.add_argument("--data_xview_dir", type=str, help="to save data files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_cls/')

    parser.add_argument("--cat_sample_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/{}/{}_cls/')

    parser.add_argument("--val_percent", type=float, default=0.20,
                        help="Percent to split into validation (ie .25 = val set is 25% total)")
    parser.add_argument("--font3", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 10}")
    parser.add_argument("-ft2", "--font2", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 13}")

    parser.add_argument("--class_num", type=int, default=1, help="Number of Total Categories")  # 60  6
    parser.add_argument("--seed", type=int, default=1024, help="random seed")
    parser.add_argument("--tile_size", type=int, default=608, help="image size")  # 300 416

    parser.add_argument("--syn_display_type", type=str, default='syn_texture',
                        help="syn_texture, syn_color, syn_mixed, syn_color0, syn_texture0, syn (match 0)")  # ######*********************change

    parser.add_argument("--resolution", type=float, default=0.3, help="resolution of synthetic data")

    syn_args = parser.parse_args()
    syn_args.data_xview_dir = syn_args.data_xview_dir.format(syn_args.class_num)
    syn_args.cat_sample_dir = syn_args.cat_sample_dir.format(syn_args.tile_size, syn_args.class_num)
    return syn_args


if __name__ == "__main__":
    syn_args = get_part_syn_args()
    folder1 = 'syn_xview_bkg_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_bxmuller_promu_size_bias0.06_RC5_v23'
    xlbl1  = folder1[folder1.find('promu_size'):]
    rgb_dir1 = syn_args.syn_images_save_dir.format(folder1)
    folder2 = 'syn_xview_bkg_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.06_bxmuller_color_bias0_RC5_v11'
    xlbl2  = folder2[folder2.find('ssig'):]
    rgb_dir2 = syn_args.syn_images_save_dir.format(folder2)
    compare_images(rgb_dir1, xlbl1,  rgb_dir2, xlbl2)

