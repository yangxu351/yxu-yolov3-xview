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
import cv2


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


def compare_images(rgb_dir1, xlbl1, rgb_dir2, xlbl2, save_dir):
    step = syn_args.tile_size * syn_args.resolution

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    rgb_dir1 = os.path.join(rgb_dir1, 'color_all_images_step{}'.format(step))
    rgb_dir2 = os.path.join(rgb_dir2, 'color_all_images_step{}'.format(step))
    print('dirs ', rgb_dir2)
    tx_image_files = np.sort(glob(os.path.join(rgb_dir1, '*.png')))
    print('len ', len(tx_image_files))
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

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    randx = np.random.uniform(-1, 1, 3) * np.array([hgain, sgain, vgain])
    x = (randx + 1).astype(np.float32)  # random gains
    img_hsv = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV) * x.reshape((1, 1, 3))).clip(None, 255).astype(np.uint8)
    # cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR), randx


def compare_images_aug_ornot(img, ori_xlbl,  aug_img, aug_xlbl, save_file):
    fig, axs = plt.subplots(1, 3, figsize=(15, 8), sharex=True, sharey=True)
    axs[0].imshow(img)
    axs[0].set_xlabel(ori_xlbl)
    axs[0].set_title(os.path.basename(save_file))
    axs[1].imshow(aug_img)
    axs[1].set_xlabel(aug_xlbl)
    axs[2].imshow(img - aug_img)
    axs[2].set_xlabel('left - right')
    plt.tight_layout()
    plt.show()
    fig.savefig(save_file)


def compare_RGB_channels(rgb_dir1, xlbl1, rgb_dir2, xlbl2, save_dir):
    step = syn_args.tile_size * syn_args.resolution

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    rgb_dir1 = os.path.join(rgb_dir1, 'color_all_images_step{}'.format(step))
    rgb_dir2 = os.path.join(rgb_dir2, 'color_all_images_step{}'.format(step))
    print('dirs ', rgb_dir2)
    tx_image_files = np.sort(glob(os.path.join(rgb_dir1, '*.png')))
    print('len ', len(tx_image_files))
    tx_img_names = [os.path.basename(f) for f in tx_image_files]

    for ix in range(len(tx_image_files)):
    # for ix in range(10):
        file1 = tx_image_files[ix]
        file2 = os.path.join(rgb_dir2, tx_img_names[ix])

        img_file1 = io.imread(file1)
        img_file2 = io.imread(file2)

        diff_img = img_file1 - img_file2

        fig, axs = plt.subplots(2, 3, figsize=(15, 8))

        axs[0, 0].imshow(img_file1)
        axs[0, 0].set_title(tx_img_names[ix])
        axs[0, 0].set_xlabel(xlbl1)
        axs[0, 1].imshow(img_file2)
        axs[0, 1].set_title(tx_img_names[ix])
        axs[0, 1].set_xlabel(xlbl2)
        axs[0, 2].imshow(diff_img)
        axs[0, 2].set_xlabel('left - right')
        # axs[1, 0].imshow(img_file1[:, :, 0] - img_file2[:, :, 0])
        axs[1, 0].hist(diff_img[:, :, 0].ravel(), range=[1, 255], bins=10)
        axs[1, 0].set_xlabel('left - right R')
        axs[1, 0].set_ylim(0, 300)
        # axs[1, 1].imshow(img_file1[:, :, 1] - img_file2[:, :, 1])
        axs[1, 1].hist(diff_img[:, :, 1].ravel(), range=[1, 255], bins=10)
        axs[1, 1].set_ylim(0, 300)
        axs[1, 1].set_xlabel('left - right G')
        # axs[1, 2].imshow(img_file1[:, :, 2] - img_file2[:, :, 2])
        axs[1, 2].hist(diff_img[:, :, 2].ravel(), range=[1, 255], bins=10)
        axs[1, 2].set_ylim(0, 300)
        axs[1, 2].set_xlabel('left - right B')
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, tx_img_names[ix]))
        # plt.show()
        plt.cla()
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
    # folder1 = 'syn_xview_bkg_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_bxmuller_promu_size_bias0.06_RC5_v23'
    # xlbl1  = folder1[folder1.find('promu_size'):]
    # rgb_dir1 = syn_args.syn_images_save_dir.format(folder1)
    # folder2 = 'syn_xview_bkg_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.06_bxmuller_color_bias0_RC5_v11'
    # xlbl2  = folder2[folder2.find('ssig'):]
    # rgb_dir2 = syn_args.syn_images_save_dir.format(folder2)
    #  save_dir = os.path.join(syn_args.cat_sample_dir, 'compare_rgb_same')
    # compare_images(rgb_dir1, xlbl1,  rgb_dir2, xlbl2, save_dir)


    '''
    adjust hsv 
    '''
    # px_thres = 23
    # whr_thres = 3
    # folder = 'syn_xview_bkg_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_bxmuller_color_bias0_RC4_v11'
    # img_dir = os.path.join(syn_args.syn_images_save_dir.format(folder), 'color_all_images_step182.4')
    # save_dir = os.path.join(syn_args.syn_images_save_dir.format(folder), 'aughsv')
    # img_files = glob(os.path.join(img_dir, '*.png'))
    # # h = 0.0138
    # # s = 0.678
    # # v = 0.36
    # h = 0.0138
    # s = 0.339
    # v = 0.18
    # np.random.seed(2)
    # for img_f in img_files:
    #     img_name = os.path.basename(img_f)
    #     img = io.imread(img_f)
    #     ori_xlbl = 'original'
    #     aug_img, hsv= augment_hsv(img, h, s, v)
    #     aug_xlbl = 'agument h={:.3f} s={:.3f} v={:.3f}'.format(hsv[0], hsv[1], hsv[2])
    #     save_file = os.path.join(save_dir, img_name)
    #     compare_images_aug_ornot(img, ori_xlbl,  aug_img, aug_xlbl, save_file)

    '''
    compare images
    plot histogram
    '''
    # img_dir = '/media/lab/Yang/data/comp_syn/'

    # size0_folder = 'syn_xview_bkg_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_promu_size_bias0_RC1_v40'
    # xlbl1 = 'size0'
    # c0_folder = 'syn_xview_bkg_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_ssig0_color_bias0_RC1_v50'
    # csig_folder = 'syn_xview_bkg_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_ssig0_color_bias45_RC1_v53'
    # xlbl = 'size0_csig45'
    # save_dir= os.path.join(syn_args.cat_sample_dir, 'compare_size0_csig0_RC1')

    # size0_folder = 'syn_xview_bkg_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_size_bias0.09_RC4_v43'
    # xlbl1 = 'size0.09'
    # c0_folder = 'syn_xview_bkg_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_bias0_RC4_v50'
    # xlbl2 = 'size0.09_csig0'
    # csig_folder = 'syn_xview_bkg_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_bias60_RC4_v54'
    # xlbl2 = 'size0.09_csig60'
    # save_dir= os.path.join(syn_args.cat_sample_dir, 'compare_RC4', xlbl2)

    # size0_folder = 'syn_xview_bkg_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_size_bias0.03_RC5_v41'
    # xlbl1 = 'size0.03'
    # c0_folder = 'syn_xview_bkg_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.03_color_bias0_RC5_v50'
    # xlbl2 = 'size0.03_csig0'
    # save_dir= os.path.join(syn_args.cat_sample_dir, 'compare_RC4', xlbl2)
    # compare_RGB_channels(os.path.join(img_dir, size0_folder), xlbl1, os.path.join(img_dir, c0_folder), xlbl2, save_dir)
    # compare_RGB_channels(os.path.join(img_dir, size0_folder), xlbl1, os.path.join(img_dir, csig_folder), xlbl2, save_dir)

    # save_dir= os.path.join(syn_args.cat_sample_dir, 'compare_size0_csig45_RC1')
    # compare_RGB_channels(os.path.join(img_dir, c0_folder), 'size0_csig0', os.path.join(img_dir, csig_folder), xlbl2, save_dir)

    '''
    histogram of RGB
    '''
    img_dir = '/media/lab/Yang/data/comp_syn/'
    folder = 'syn_xview_bkg_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_bias0_RC4_v50'
    files = np.sort(glob(os.path.join(img_dir, folder, 'color_all_images_step182.4', '*.png')))
    folder2 = 'syn_xview_bkg_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_bias15_RC4_v51'
    files2 = np.sort(glob(os.path.join(img_dir, folder2, 'color_all_images_step182.4', '*.png')))
    save_dir = os.path.join(syn_args.cat_sample_dir, 'compare_RC4', 'comp_0_15')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for ix, f in enumerate(files):
        img = io.imread(f)
        img2 = io.imread(files2[ix])
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        axs[0, 0].imshow(img)
        axs[0, 0].set_title(os.path.basename(f))
        # axs[0, 0].set_xlabel('s')
        axs[0, 1].imshow(img2)
        axs[0, 2].imshow(img-img2)
        axs[1, 0].hist(img[:, :, 0].ravel(), bins = 30)
        axs[1, 1].hist(img[:, :, 1].ravel(), bins = 30)
        axs[1, 2].hist(img[:, :, 2].ravel(), bins = 30)
        axs[1, 0].hist(img2[:, :, 0].ravel(), bins = 30)
        axs[1, 1].hist(img2[:, :, 1].ravel(), bins = 30)
        axs[1, 2].hist(img2[:, :, 2].ravel(), bins = 30)
        plt.show()

        # plt.savefig(os.path.joint(save_dir, os.path.basename(f)))
        # fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        # axs[0, 0].hist(img[:, :, 0].ravel(), bins = 30)
        # axs[0, 1].hist(img[:, :, 1].ravel(), bins = 30)
        # axs[0, 2].hist(img[:, :, 2].ravel(), bins = 30)
        # axs[0, 0].hist(img2[:, :, 0].ravel(), bins = 30)
        # axs[0, 1].hist(img2[:, :, 1].ravel(), bins = 30)
        # axs[0, 2].hist(img2[:, :, 2].ravel(), bins = 30)
        # axs[1, 0].hist(img2[:, :, 0].ravel(), bins = 30)
        # axs[1, 1].hist(img2[:, :, 1].ravel(), bins = 30)
        # axs[1, 2].hist(img2[:, :, 2].ravel(), bins = 30)
        # plt.show()






