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
from utils.xview_synthetic_util import preprocess_xview_syn_data_distribution as pps

def compare_overlap():
    # file1 = '/media/lab/Yang/data/synthetic_data/Airplanes/syn_radial_200_images_test_ts608/airplanes_radial_200_59_RGB.jpg'
    # file2 = '/media/lab/Yang/data/synthetic_data/Airplanes/syn_radial_200_images_test_ts608/airplanes_radial_200_60_RGB.jpg'
    file1 = '/media/lab/Yang/data/synthetic_data/Airplanes/syn_radial_200_images_test_ts608/airplanes_radial_200_75_RGB.jpg'
    file2 = '/media/lab/Yang/data/synthetic_data/Airplanes/syn_radial_200_images_test_ts608/airplanes_radial_200_76_RGB.jpg'

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

    plt.savefig('/media/lab/Yang/data/synthetic_data/Airplanes/75_76common_change.jpg')


def compare_images_with_different_display_type(two=False):
    syn_args = pps.get_syn_args()
    cities = ['barcelona', 'berlin', 'francisco', 'hexagon', 'radial', 'siena', 'spiral']
    streets = [200, 200, 200, 200, 200, 250, 130]
    syn_plane_img_anno_dir = '/media/lab/Yang/data/synthetic_data/Airplanes/{}'
    city = cities[0]
    sts = streets[0]

    step = syn_args.tile_size * syn_args.resolution

    if two:
        '''
        first --> 0 
        '''
        IMG_FORMAT = '.jpg'
        save_dir = os.path.join('/media/lab/Yang/data/synthetic_data/Airplanes/', 'check_rgb_gt0', city)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        tx_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture0'), '{}_{}_{}_images_step{}'.format('syn_texture0', city, sts, step))
        tx_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture0'), '{}_{}_{}_annos_step{}'.format('syn_texture0',  city, sts, step))

        clr_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color0'), '{}_{}_{}_images_step{}'.format('syn_color0', city, sts, step))
        clr_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color0'), '{}_{}_{}_annos_step{}'.format('syn_color0',  city, sts, step))
        tx_image_files = np.sort(glob(os.path.join(tx_img_dir, '*' + IMG_FORMAT)))
        tx_img_names = [os.path.basename(f) for f in tx_image_files]
        clr_image_files = np.sort(glob(os.path.join(clr_img_dir, '*' + IMG_FORMAT)))
        clr_img_names = [os.path.basename(f) for f in clr_image_files]

        tx_lbl_files = np.sort(glob(os.path.join(tx_lbl_dir, '*' + IMG_FORMAT)))
        # tx_lbl_names = [os.path.basename(f) for f in tx_lbl_files]
        clr_lbl_files = np.sort(glob(os.path.join(clr_lbl_dir, '*' + IMG_FORMAT)))
        # clr_lbl_names = [os.path.basename(f) for f in clr_lbl_files]

        for ix in range(len(tx_image_files)):
            file1 = tx_image_files[ix]
            file2 = os.path.join(clr_img_dir, clr_img_names[ix])

            img_file1 = io.imread(file1)
            img_file2 = io.imread(file2)

            file21 = tx_lbl_files[ix]
            file22 = clr_lbl_files[ix]

            img_file21 = io.imread(file21)
            img_file22 = io.imread(file22)

            fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

            axs[0, 0].imshow(img_file1)
            axs[0, 0].set_title('texture')
            axs[0, 1].imshow(img_file2)
            axs[0, 1].set_title('color')
            axs[1, 0].imshow(img_file21)
            axs[1, 1].imshow(img_file22)
            axs[1, 1].imshow(img_file21- img_file22)
            axs[1, 0].set_xlabel(tx_img_names[ix])
            axs[1, 2].set_xlabel(clr_img_names[ix])
            axs[1, 2].set_xlabel('left - right')
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, tx_img_names[ix].replace('texture_', '')))
            # plt.show()
    else:
        '''
        second 
        '''
        IMG_FORMAT = '.png'
        save_dir = os.path.join('/media/lab/Yang/data/synthetic_data/Airplanes/', 'check_rgb_gt', city)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        tx_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture'), '{}_{}_{}_images_step{}'.format('syn_texture', city, sts, step))
        tx_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture'), '{}_{}_{}_annos_step{}'.format('syn_texture',  city, sts, step))

        clr_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color'), '{}_{}_{}_images_step{}'.format('syn_color', city, sts, step))
        clr_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color'), '{}_{}_{}_annos_step{}'.format('syn_color',  city, sts, step))

        mx_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_mixed'), '{}_{}_{}_images_step{}'.format('syn_mixed', city, sts, step))
        mx_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_mixed'), '{}_{}_{}_annos_step{}'.format('syn_mixed',  city, sts, step))

        tx_image_files = np.sort(glob(os.path.join(tx_img_dir, '*' + IMG_FORMAT)))
        tx_img_names = [os.path.basename(f) for f in tx_image_files]
        clr_image_files = np.sort(glob(os.path.join(clr_img_dir, '*' + IMG_FORMAT)))
        clr_img_names = [os.path.basename(f) for f in clr_image_files]
        mx_image_files = np.sort(glob(os.path.join(mx_img_dir, '*' + IMG_FORMAT)))
        mx_img_names = [os.path.basename(f) for f in mx_image_files]

        tx_lbl_files = np.sort(glob(os.path.join(tx_lbl_dir, '*' + IMG_FORMAT)))
        # tx_lbl_names = [os.path.basename(f) for f in tx_lbl_files]
        clr_lbl_files = np.sort(glob(os.path.join(clr_lbl_dir, '*' + IMG_FORMAT)))
        # clr_lbl_names = [os.path.basename(f) for f in clr_lbl_files]
        mx_lbl_files = np.sort(glob(os.path.join(mx_lbl_dir, '*' + IMG_FORMAT)))
        # mx_lbl_names = [os.path.basename(f) for f in mx_lbl_files]

        for ix in range(len(tx_image_files)):
            file1 = tx_image_files[ix]
            file2 = os.path.join(clr_img_dir, clr_img_names[ix])
            file3 = os.path.join(mx_img_dir, mx_img_names[ix])

            img_file1 = io.imread(file1)
            img_file2 = io.imread(file2)
            img_file3 = io.imread(file3)

            file21 = tx_lbl_files[ix]
            file22 = clr_lbl_files[ix]
            file23 = mx_lbl_files[ix]

            img_file21 = io.imread(file21)
            img_file22 = io.imread(file22)
            img_file23 = io.imread(file23)

            # if np.all(img_file3==img_file2) and np.all(img_file2==img_file1):
            #     continue

            # print(tx_image_files[ix])

            fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

            axs[0, 0].imshow(img_file1)
            axs[0, 0].set_title('texture')
            axs[0, 1].imshow(img_file2)
            axs[0, 1].set_title('color')
            axs[0, 2].imshow(img_file3)
            axs[0, 2].set_title('mixed')
            axs[1, 0].imshow(img_file21)
            axs[1, 1].imshow(img_file22)
            axs[1, 2].imshow(img_file23)
            axs[1, 0].set_xlabel(tx_img_names[ix])
            axs[1, 1].set_xlabel(clr_img_names[ix])
            axs[1, 2].set_xlabel(mx_img_names[ix])
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, tx_img_names[ix].replace('texture_', '')))
            # plt.show()


def compare_images_after_combine(two=False):

    '''
    all images annos
    '''
    syn_args = pps.get_syn_args()
    syn_plane_img_anno_dir = '/media/lab/Yang/data/synthetic_data/Airplanes/{}'

    step = syn_args.tile_size * syn_args.resolution

    if two:
        '''
        first --> 0 
        '''
        IMG_FORMAT = '.jpg'
        save_dir = os.path.join('/media/lab/Yang/data/synthetic_data/Airplanes/', 'check_rgb_gt0', 'all')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            shutil.rmtree(save_dir)
            os.makedirs(save_dir)
        tx_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture0'), '{}_all_images_step{}'.format('syn_texture0', step))
        tx_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture0'), '{}_all_annos_step{}'.format('syn_texture0', step))

        clr_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color0'), '{}_all_images_step{}'.format('syn_color0', step))
        clr_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color0'), '{}_all_annos_step{}'.format('syn_color0', step))

        tx_image_files = np.sort(glob(os.path.join(tx_img_dir, '*' + IMG_FORMAT)))
        tx_img_names = [os.path.basename(f) for f in tx_image_files]
        clr_image_files = np.sort(glob(os.path.join(clr_img_dir, '*' + IMG_FORMAT)))
        clr_img_names = [os.path.basename(f) for f in clr_image_files]

        tx_lbl_files = np.sort(glob(os.path.join(tx_lbl_dir, '*' + IMG_FORMAT)))
        # tx_lbl_names = [os.path.basename(f) for f in tx_lbl_files]
        clr_lbl_files = np.sort(glob(os.path.join(clr_lbl_dir, '*' + IMG_FORMAT)))
        # clr_lbl_names = [os.path.basename(f) for f in clr_lbl_files]

        for ix in range(len(tx_image_files)):
            file1 = tx_image_files[ix]
            file2 = os.path.join(clr_img_dir, clr_img_names[ix])

            img_file1 = io.imread(file1)
            img_file2 = io.imread(file2)

            file21 = tx_lbl_files[ix]
            file22 = clr_lbl_files[ix]

            img_file21 = io.imread(file21)
            img_file22 = io.imread(file22)

            fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

            axs[0, 0].imshow(img_file1)
            axs[0, 0].set_title('texture')
            axs[0, 1].imshow(img_file2)
            axs[0, 1].set_title('color')
            axs[1, 0].imshow(img_file21)
            axs[1, 1].imshow(img_file22)
            axs[1, 0].set_xlabel(tx_img_names[ix])
            axs[1, 1].set_xlabel(clr_img_names[ix])
            axs[1, 2].imshow(img_file21- img_file22)
            axs[1, 2].set_xlabel('left - right')
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, tx_img_names[ix].replace('texture_', '')))
            # plt.show()
    else:

        IMG_FORMAT = '.png'
        save_dir = os.path.join('/media/lab/Yang/data/synthetic_data/Airplanes/', 'check_rgb_gt0', 'all')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        tx_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture'), '{}_all_images_step{}'.format('syn_texture', step))
        tx_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture'), '{}_all_annos_step{}'.format('syn_texture', step))

        clr_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color'), '{}_all_images_step{}'.format('syn_color', step))
        clr_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color'), '{}_all_annos_step{}'.format('syn_color', step))

        mx_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_mixed'), '{}_all_images_step{}'.format('syn_mixed', step))
        mx_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_mixed'), '{}_all_annos_step{}'.format('syn_mixed', step))
        tx_image_files = np.sort(glob(os.path.join(tx_img_dir, '*' + IMG_FORMAT)))
        tx_img_names = [os.path.basename(f) for f in tx_image_files]
        clr_image_files = np.sort(glob(os.path.join(clr_img_dir, '*' + IMG_FORMAT)))
        clr_img_names = [os.path.basename(f) for f in clr_image_files]
        mx_image_files = np.sort(glob(os.path.join(mx_img_dir, '*' + IMG_FORMAT)))
        mx_img_names = [os.path.basename(f) for f in mx_image_files]

        tx_lbl_files = np.sort(glob(os.path.join(tx_lbl_dir, '*' + IMG_FORMAT)))
        # tx_lbl_names = [os.path.basename(f) for f in tx_lbl_files]
        clr_lbl_files = np.sort(glob(os.path.join(clr_lbl_dir, '*' + IMG_FORMAT)))
        # clr_lbl_names = [os.path.basename(f) for f in clr_lbl_files]
        mx_lbl_files = np.sort(glob(os.path.join(mx_lbl_dir, '*' + IMG_FORMAT)))
        # mx_lbl_names = [os.path.basename(f) for f in mx_lbl_files]
        for ix in range(len(tx_image_files)):
            file1 = tx_image_files[ix]
            file2 = os.path.join(clr_img_dir, clr_img_names[ix])
            file3 = os.path.join(mx_img_dir, mx_img_names[ix])

            img_file1 = io.imread(file1)
            img_file2 = io.imread(file2)
            img_file3 = io.imread(file3)

            file21 = tx_lbl_files[ix]
            file22 = clr_lbl_files[ix]
            file23 = mx_lbl_files[ix]

            img_file21 = io.imread(file21)
            img_file22 = io.imread(file22)
            img_file23 = io.imread(file23)

            fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

            axs[0, 0].imshow(img_file1)
            axs[0, 0].set_title('texture')
            axs[0, 1].imshow(img_file2)
            axs[0, 1].set_title('color')
            axs[0, 2].imshow(img_file3)
            axs[0, 2].set_title('mixed')
            axs[1, 0].imshow(img_file21)
            axs[1, 1].imshow(img_file22)
            axs[1, 2].imshow(img_file23)
            axs[1, 0].set_xlabel(tx_img_names[ix])
            axs[1, 1].set_xlabel(clr_img_names[ix])
            axs[1, 2].set_xlabel(mx_img_names[ix])
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, tx_img_names[ix].replace('texture_', '')))
            # plt.show()


def compare_first_second_dataset():
    syn_args = pps.get_syn_args()
    syn_plane_img_anno_dir = '/media/lab/Yang/data/synthetic_data/Airplanes/{}'
    save_dir = os.path.join('/media/lab/Yang/data/synthetic_data/Airplanes/', 'comp_first_sec_rgb_gt', 'rgb')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    step = syn_args.tile_size * syn_args.resolution

    IMG_FORMAT0 = '.jpg'
    tx0_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture0'), '{}_all_images_step{}'.format('syn_texture0', step))
    tx0_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture0'), '{}_all_annos_step{}'.format('syn_texture0', step))

    clr0_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color0'), '{}_all_images_step{}'.format('syn_color0', step))
    clr0_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color0'), '{}_all_annos_step{}'.format('syn_color0', step))

    tx0_image_files = np.sort(glob(os.path.join(tx0_img_dir, '*' + IMG_FORMAT0)))
    tx0_img_names = [os.path.basename(f) for f in tx0_image_files]
    clr0_image_files = np.sort(glob(os.path.join(clr0_img_dir, '*' + IMG_FORMAT0)))
    clr0_img_names = [os.path.basename(f) for f in clr0_image_files]

    # tx0_lbl_files = np.sort(glob(os.path.join(tx0_lbl_dir, '*' + IMG_FORMAT0)))
    # # tx0_lbl_names = [os.path.basename(f) for f in tx_lbl_files]
    # clr0_lbl_files = np.sort(glob(os.path.join(clr0_lbl_dir, '*' + IMG_FORMAT0)))

    IMG_FORMAT = '.png'
    tx_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture'), '{}_all_images_step{}'.format('syn_texture', step))
    tx_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture'), '{}_all_annos_step{}'.format('syn_texture', step))

    clr_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color'), '{}_all_images_step{}'.format('syn_color', step))
    clr_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color'), '{}_all_annos_step{}'.format('syn_color', step))
    tx_image_files = np.sort(glob(os.path.join(tx_img_dir, '*' + IMG_FORMAT)))
    tx_img_names = [os.path.basename(f) for f in tx_image_files]
    clr_image_files = np.sort(glob(os.path.join(clr_img_dir, '*' + IMG_FORMAT)))
    clr_img_names = [os.path.basename(f) for f in clr_image_files]

    # tx_lbl_files = np.sort(glob(os.path.join(tx_lbl_dir, '*' + IMG_FORMAT)))
    # # tx_lbl_names = [os.path.basename(f) for f in tx_lbl_files]
    # clr_lbl_files = np.sort(glob(os.path.join(clr_lbl_dir, '*' + IMG_FORMAT)))
    # # clr_lbl_names = [os.path.basename(f) for f in clr_lbl_files]
    tx_img_not_in_tx0 = [f for f in tx_img_names if f.replace(IMG_FORMAT, IMG_FORMAT0) not in tx0_img_names]
    tx0_img_not_in_tx = [f for f in tx0_img_names if f.replace(IMG_FORMAT0, IMG_FORMAT) not in tx_img_names]
    print('tx_img_not_in_tx0', len(tx_img_not_in_tx0), tx_img_not_in_tx0)
    print('tx0_img_not_in_tx', len(tx0_img_not_in_tx), tx0_img_not_in_tx)
    for ix in range(len(tx0_img_names)):
        if tx0_img_names[ix] in tx0_img_not_in_tx:
            continue
        txf0 = os.path.join(tx0_img_dir, tx0_img_names[ix])
        clf0 = os.path.join(clr0_img_dir, clr0_img_names[ix])
        txf = os.path.join(tx_img_dir, tx0_img_names[ix].replace(IMG_FORMAT0, IMG_FORMAT))
        clf = os.path.join(clr_img_dir, clr0_img_names[ix].replace(IMG_FORMAT0, IMG_FORMAT))

        img_file01 = io.imread(txf0)
        img_file02 = io.imread(clf0)
        img_file1 = io.imread(txf)
        img_file2 = io.imread(clf)

        fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

        axs[0, 0].imshow(img_file01)
        axs[0, 0].set_title('syn_texture0')
        axs[1, 0].imshow(img_file02)
        axs[1, 0].set_title('syn_color0')
        axs[0, 1].imshow(img_file1)
        axs[0, 1].set_title('syn_texture')
        axs[1, 1].imshow(img_file2)
        axs[1, 1].set_title('syn_color')
        axs[0, 2].imshow(img_file01 - img_file1)
        axs[0, 2].set_title('syn_texture0 - syn_texture')
        axs[1, 2].imshow(img_file01 - img_file1)
        axs[1, 2].set_title('syn_color0 - syn_color')
        axs[0, 0].set_ylabel(tx0_img_names[ix])
        axs[1, 0].set_ylabel(clr0_img_names[ix])
        plt.tight_layout()
        # plt.show()
        fig.savefig(os.path.join(save_dir, tx0_img_names[ix].replace('texture_', '')))

    tx_save_dir = os.path.join('/media/lab/Yang/data/synthetic_data/Airplanes/', 'comp_first_sec_rgb_gt', 'tx_img_not_in_tx0')
    if not os.path.exists(tx_save_dir):
        os.makedirs(tx_save_dir)
    for tx in tx_img_not_in_tx0:
        txf = os.path.join(tx_img_dir, tx)
        clf = os.path.join(clr_img_dir, tx.replace('texture_', 'color_'))
        tx_lbl = os.path.join(tx_lbl_dir, tx)
        clf_lbl = os.path.join(clr_lbl_dir, tx.replace('texture_', 'color_'))
        img_file1 = io.imread(txf)
        img_file2 = io.imread(clf)
        img_file11 = io.imread(tx_lbl)
        img_file12 = io.imread(clf_lbl)
        fig, axs = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)

        axs[0, 0].imshow(img_file1)
        axs[0, 0].set_title('syn_texture')
        axs[0, 1].imshow(img_file2)
        axs[0, 1].set_title('syn_color')
        axs[1, 0].imshow(img_file11)
        # axs[1, 0].set_title('syn_texture')
        axs[1, 1].imshow(img_file12)
        # axs[1, 1].set_title('syn_color')
        axs[1, 0].set_xlabel(tx)
        axs[1, 1].set_xlabel(tx.replace('texture_', 'color_'))
        plt.tight_layout()
        fig.savefig(os.path.join(tx_save_dir, tx.replace('texture_', '')))

    tx0_save_dir = os.path.join('/media/lab/Yang/data/synthetic_data/Airplanes/', 'comp_first_sec_rgb_gt', 'tx0_img_not_in_tx')
    if not os.path.exists(tx0_save_dir):
        os.makedirs(tx0_save_dir)
    for tx0 in tx0_img_not_in_tx:
        txf0 = os.path.join(tx0_img_dir, tx0)
        clf0 = os.path.join(clr0_img_dir, tx0.replace('texture_', 'color_'))
        txf0_lbl = os.path.join(tx0_lbl_dir, tx0)
        clf0_lbl = os.path.join(clr0_lbl_dir, tx0.replace('texture_', 'color_'))
        img_file01 = io.imread(txf0)
        img_file02 = io.imread(clf0)
        img_file11 = io.imread(txf0_lbl)
        img_file12 = io.imread(clf0_lbl)

        fig, axs = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)

        axs[0, 0].imshow(img_file01)
        axs[0, 0].set_title('syn_texture')
        axs[0, 1].imshow(img_file02)
        axs[0, 1].set_title('syn_color')
        axs[1, 0].imshow(img_file11)
        # axs[0, 0].set_title('syn_texture')
        axs[1, 1].imshow(img_file12)
        # axs[0, 1].set_title('syn_color')
        axs[1, 0].set_xlabel(tx0)
        axs[1, 1].set_xlabel(tx0.replace('texture_', 'color_'))
        plt.tight_layout()
        fig.savefig(os.path.join(tx0_save_dir, tx0.replace('texture_', '')))


def check_input_files_names():
    base_dir = '/data/users/yang/code/yxu-yolov3-xview' 
    file_dir = os.path.join(base_dir, 'input_trn_files')
    val_labeled_files = np.sort(glob(os.path.join(file_dir, '*_val_labeled_trn*.json')))
    val_labeled_miss_files = np.sort(glob(os.path.join(file_dir, '*miss*.json')))
    val_xview_files = np.sort(glob(os.path.join(file_dir, '*val_xview*.json')))
    val_syn_files = np.sort(glob(os.path.join(file_dir, '*val_syn*.json')))
    print('xview_labeled_files', val_labeled_files)
    print('xview_xview_files', val_xview_files)
    color_labeled_maps = json.load(open(val_labeled_files[0]))
    mixed_labeled_maps = json.load(open(val_labeled_files[1]))
    color_miss_maps = json.load(open(val_labeled_miss_files[0]))
    mixed_miss_maps = json.load(open(val_labeled_miss_files[1]))
    color_xview_maps = json.load(open(val_xview_files[0]))
    mixed_xview_maps = json.load(open(val_xview_files[1]))
    color_syn_maps = json.load(open(val_syn_files[0]))
    mixed_syn_maps = json.load(open(val_syn_files[1]))
    color_maps = mixed_labeled_maps
    mixed_maps = mixed_xview_maps
    print('color_maps[0]', color_maps['0'])
    for i in range(20):
        cl_files = [c.split('color_')[-1] for c in color_maps[str(i)]]
        ml_files = [m.split('mixed_')[-1] for m in mixed_maps[str(i)]]
        print(cl_files[0], ml_files[0])
        if cl_files == ml_files:
            print('true')
        else:
            print('false')
    for i in range(20):
        cl_files = [c.split('mixed_')[-1] for c in color_maps[str(i)]]
        ml_files = [m.split('mixed_')[-1] for m in mixed_maps[str(i)]]
        print(cl_files[0], ml_files[0])
        if cl_files == ml_files:
            print('true')
        else:
            print('false')


def get_model_hash(model_dict):
    model_dict_vlu = [v for v in model_dict.values()]
    model_dict_vlu_np = [a.cpu().data.numpy() for a in model_dict_vlu]
    model_dict_vlu_np_lst = [a.tolist() for a in model_dict_vlu_np]
    return hash(json.dumps(model_dict_vlu_np_lst))   


def check_model_hash():
    import torch
    base_dir = '/data/users/yang/code/yxu-yolov3-xview' 
    mixed_file_dir = os.path.join(base_dir, 'weights/1_cls/syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_model4_v7_mixed_seed17')
    color_file_dir = os.path.join(base_dir, 'weights/1_cls/syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_model4_v7_color_seed17')
    save_color_hash_dir = os.path.join(base_dir, color_file_dir, 'hash_folder')
#    if not os.path.exists(save_color_hash_dir):
#        os.makedirs(save_color_hash_dir)
    save_mixed_hash_dir = os.path.join(base_dir, mixed_file_dir, 'hash_folder')
#    if not os.path.exists(save_mixed_hash_dir):
#        os.makedirs(save_mixed_hash_dir)
   
    color_hash_maps = {}
    mixed_hash_maps = {}
    val_cmts = ['val_syn', 'val_labeled_seed', 'val_labeled_miss']
    for cmt in val_cmts:
        mixed_files = np.sort(glob(os.path.join(mixed_file_dir, '2020-06-09*_{}*/best_*.pt'.format(cmt))))
        mixed_pt = torch.load(mixed_files[0])[215]['model']
        mixed_hash = get_model_hash(mixed_pt)
        mixed_txt = open(os.path.join(save_mixed_hash_dir, '{}_hash.txt'.format(cmt)), 'w')
        mixed_txt.write('{}\n'.format(mixed_hash))
        mixed_txt.close()
        mixed_hash_maps[cmt] = mixed_hash

        color_files = np.sort(glob(os.path.join(color_file_dir, '2020-06-09*_{}*/best_*.pt'.format(cmt))))
        color_pt = torch.load(color_files[0])[215]['model']
        color_hash = get_model_hash(color_pt)
        color_txt = open(os.path.join(save_color_hash_dir, '{}_hash.txt'.format(cmt)), 'w')
        color_txt.write('{}\n'.format(color_hash))
        color_txt.close()
        color_hash_maps[cmt] = color_hash
    color_json_file = os.path.join(save_color_hash_dir, 'color_hash_maps.json')
    json.dump(color_hash_maps, open(color_json_file, 'w'), ensure_ascii=False, indent=2)
    mixed_json_file = os.path.join(save_mixed_hash_dir, 'mixed_hash_maps.json')
    json.dump(mixed_hash_maps, open(mixed_json_file, 'w'), ensure_ascii=False, indent=2) 
    

def check_model_hash_before_val_after():
    import torch
    base_dir = '/data/users/yang/code/yxu-yolov3-xview' 
    mixed_file_dir = os.path.join(base_dir, 'weights/1_cls/syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_model4_v7_mixed_seed17')
    color_file_dir = os.path.join(base_dir, 'weights/1_cls/syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_model4_v7_color_seed17')
#    save_color_hash_dir = os.path.join(base_dir, color_file_dir, 'hash_folder_06-10_11')
    save_color_hash_dir = os.path.join(base_dir, color_file_dir, 'hash_folder_06-10_22')
    if not os.path.exists(save_color_hash_dir):
        os.makedirs(save_color_hash_dir)
#    save_mixed_hash_dir = os.path.join(base_dir, mixed_file_dir, 'hash_folder_06-10_11')
    save_mixed_hash_dir = os.path.join(base_dir, mixed_file_dir, 'hash_folder_06-10_22')
    if not os.path.exists(save_mixed_hash_dir):
        os.makedirs(save_mixed_hash_dir)
   
    color_hash_maps = {}
    mixed_hash_maps = {}
    val_cmts = ['val_syn',  'val_xview', 'val_labeled_seed', 'val_labeled_miss']
    for cmt in val_cmts:
#        mixed_files = np.sort(glob(os.path.join(mixed_file_dir, '2020-06-10_11*_{}*/last_*.pt'.format(cmt))))
        mixed_files = np.sort(glob(os.path.join(mixed_file_dir, '2020-06-10_22*_{}*/last_*.pt'.format(cmt))))
        mixed_pt = torch.load(mixed_files[0])['model']
        mixed_hash = get_model_hash(mixed_pt)
#        mixed_before_files = np.sort(glob(os.path.join(mixed_file_dir, '2020-06-10_11*_{}*/last_*before*.pt'.format(cmt))))
        mixed_before_files = np.sort(glob(os.path.join(mixed_file_dir, '2020-06-10_22*_{}*/last_*before*.pt'.format(cmt))))
        mixed_before_pt = torch.load(mixed_before_files[0])['model']
        mixed_before_hash = get_model_hash(mixed_before_pt)
        mixed_txt = open(os.path.join(save_mixed_hash_dir, '{}_hash.txt'.format(cmt)), 'w')
        mixed_txt.write('before: {}\n'.format(mixed_before_hash))
        mixed_txt.write('after: {}\n'.format(mixed_hash))
        mixed_txt.close()
        mixed_hash_maps[cmt] = [mixed_before_hash, mixed_hash]

#        color_files = np.sort(glob(os.path.join(color_file_dir, '2020-06-10_11*_{}*/last_*.pt'.format(cmt))))
        color_files = np.sort(glob(os.path.join(color_file_dir, '2020-06-10_22*_{}*/last_*.pt'.format(cmt))))
        color_pt = torch.load(color_files[0])['model']
        color_hash = get_model_hash(color_pt)
#        color_before_files = np.sort(glob(os.path.join(color_file_dir, '2020-06-10_11*_{}*/last_*before*.pt'.format(cmt))))
        color_before_files = np.sort(glob(os.path.join(color_file_dir, '2020-06-10_22*_{}*/last_*before*.pt'.format(cmt))))
        color_before_pt = torch.load(color_before_files[0])['model']
        color_before_hash = get_model_hash(color_before_pt)
        color_txt = open(os.path.join(save_color_hash_dir, '{}_hash.txt'.format(cmt)), 'w')
        color_txt.write('before: {}\n'.format(color_before_hash))
        color_txt.write('after: {}\n'.format(color_hash))
        color_txt.close()
        color_hash_maps[cmt] = [color_before_hash,color_hash]
    color_json_file = os.path.join(save_color_hash_dir, 'color_hash_maps.json')
    json.dump(color_hash_maps, open(color_json_file, 'w'), ensure_ascii=False, indent=2)
    mixed_json_file = os.path.join(save_mixed_hash_dir, 'mixed_hash_maps.json')
    json.dump(mixed_hash_maps, open(mixed_json_file, 'w'), ensure_ascii=False, indent=2) 



def check_test_files():    
    import pandas as pd
    base_dir = '/data/users/yang/code/yxu-yolov3-xview/data_xview/1_cls/px23whr3_seed17/' 
    test_xview_img = pd.read_csv(os.path.join(base_dir, 'xviewval_img_px23whr3_seed17.txt'), header=None)
    test_xview_lbl = pd.read_csv(os.path.join(base_dir, 'xviewval_lbl_px23whr3_seed17.txt'), header=None)
    test_labeled_img = pd.read_csv(os.path.join(base_dir, 'xviewtest_img_px23whr3_seed17_m4_labeled.txt'), header=None)
    test_labeled_lbl = pd.read_csv(os.path.join(base_dir, 'xviewtest_lbl_px23whr3_seed17_m4_labeled.txt'), header=None)
    test_miss_img = pd.read_csv(os.path.join(base_dir, 'xviewtest_img_px23whr3_seed17_m4_labeled_miss.txt'), header=None)
    test_miss_lbl = pd.read_csv(os.path.join(base_dir, 'xviewtest_lbl_px23whr3_seed17_m4_labeled_miss.txt'), header=None)
    
    xv_img_names = [os.path.basename(f).split('.')[0] for f in test_xview_img.loc[:, 0]]
    xv_lbl_names = [os.path.basename(f).split('.')[0] for f in test_xview_lbl.loc[:, 0]]
    if xv_img_names == xv_lbl_names:
        print('xv_names', True)
    labeled_img_names = [os.path.basename(f).split('.')[0] for f in test_labeled_img.loc[:, 0]]
    labeled_lbl_names = [os.path.basename(f).split('.')[0] for f in test_labeled_lbl.loc[:, 0]]
    if labeled_img_names == labeled_lbl_names:
        print('labeled_names', True)
    miss_img_names = [os.path.basename(f).split('.')[0] for f in test_miss_img.loc[:, 0]]
    miss_lbl_names = [os.path.basename(f).split('.')[0] for f in test_miss_lbl.loc[:, 0]]
    if miss_img_names == miss_lbl_names:
        print('miss_names', True)      

    if miss_lbl_names == labeled_lbl_names:
        print('labeled=miss', True)
    xv_lbl_names.sort()
    labeled_lbl_names.sort()
    if xv_lbl_names == labeled_lbl_names:
        print('xv==labeled', True)
    else:
        diff_in_labeled = [f for f in xv_lbl_names if f not in labeled_lbl_names]
        print(diff_in_labeled)
        diff_in_xv = [f for f in labeled_lbl_names if f not in xv_lbl_names]
        print(diff_in_xv)
        print('xv', xv_lbl_names)
        print('labeled', labeled_lbl_names)


if __name__ == "__main__":
    # display_type = ['syn_texture', 'syn_color', 'syn_mixed']
    # # display_type = ['syn_texture0', 'syn_color0']
    # city = ['barcelona', 'berlin', 'francisco', 'hexagon', 'radial', 'siena', 'spiral']
    # streets = [200, 200, 200, 200, 200, 250, 130]

    # two = True
    # compare_images_separate_cities(two)

    # two = True
    # compare_images_after_combine(two)

    # compare_first_second_dataset()

    '''
    check the different data selected in dataset1 and dataset2
    '''
#    tex0_files = '/media/lab/Yang/code/yolov3/data_xview/1_cls/first_data_set_backup/xview_syn_texture_0.25_train_lbl.txt'
#    # tex1_files = '/media/lab/Yang/code/yolov3/data_xview/1_cls/xview_syn_texture0_0.25/xview_syn_texture0_0.25_train_lbl_px4whr3.txt'
#    tex1_files = '/media/lab/Yang/code/yolov3/data_xview/1_cls/xview_syn_texture_0.25/xview_syn_texture_0.25_train_lbl.txt'
#    df_tex0 = pd.read_csv(tex0_files, header=None)
#    df_tex1 = pd.read_csv(tex1_files, header=None)
#    tex0_names = [os.path.basename(f) for f in df_tex0.loc[:, 0]]
#    tex1_names = [os.path.basename(f) for f in df_tex1.loc[:, 0]]
#    if np.all(tex0_names == tex1_names):
#        print(True)
#    else:
#        print(False)


    '''
    check input files sequences
    '''
#    check_input_file_names()
    
    '''
    check *.pt
    model hash
    '''
#    check_model_hash()
    check_model_hash_before_val_after()

    '''
    check test file names
    '''
#    check_test_files()

