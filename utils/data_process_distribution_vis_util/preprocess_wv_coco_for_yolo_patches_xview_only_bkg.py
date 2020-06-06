import glob
import numpy as np
import argparse
import os

import utils.wv_util as wv
from utils.utils_xview import coord_iou, compute_iou
from utils.xview_synthetic_util import preprocess_xview_syn_data_distribution as pps
from utils.data_process_distribution_vis_util import process_wv_coco_for_yolo_patches_no_trnval as pwv
import pandas as pd
from ast import literal_eval
import json
import datetime
from matplotlib import pyplot as plt
import shutil
import cv2
from tqdm import tqdm
import json
import time



def get_xview_bkg_and_lbl_files_create_realated_list_files():
    args = get_args()
    src_all_dir = args.images_save_dir.split('_{}cls'.format(args.class_num)) [0]
    all_images = glob.glob(os.path.join(src_all_dir, '*.jpg'))
    all_img_names = [os.path.basename(f) for f in all_images]

    airc_dir = args.images_save_dir
    airc_images = glob.glob(os.path.join(airc_dir, '*.jpg'))
    airc_img_names = [os.path.basename(f) for f in airc_images]
    airc_img_prefixes = [x.split('_')[0] for x in airc_img_names]
    airc_img_prefixes = list(set(airc_img_prefixes))

    bkg_img_names = [n for n in all_img_names if n.split('_')[0] not in airc_img_prefixes]

    np.random.seed(args.seed)
    bkg_indices = np.random.permutation(len(bkg_img_names))

    save_bkg_img_dir = src_all_dir + '_xview_bkg/'
    if not os.path.exists(save_bkg_img_dir):
        os.mkdir(save_bkg_img_dir)

    list_dir = os.path.join(args.data_save_dir, 'xview_bkg_only_seed{}/'.format(args.seed))
    if not os.path.exists(list_dir):
        os.mkdir(list_dir)
    xview_bkg_img_file = open(os.path.join(list_dir, 'xview_bkg_img_seed{}.txt'.format(args.seed)), 'w')
    xview_bkg_lbl_file = open(os.path.join(list_dir, 'xview_bkg_lbl_seed{}.txt'.format(args.seed)), 'w')

    save_bkg_lbl_dir = args.annos_save_dir[:-1] + '_bkg/'
    if not os.path.exists(save_bkg_lbl_dir):
        os.mkdir(save_bkg_lbl_dir)

    num_airc_imgs = len(airc_img_names)
    for i in range(num_airc_imgs):
        idx = bkg_indices[i]
        shutil.copy(os.path.join(src_all_dir, bkg_img_names[idx]),
                    os.path.join(save_bkg_img_dir, bkg_img_names[idx]))
        lbl_file = open(os.path.join(save_bkg_lbl_dir, bkg_img_names[idx].replace('.jpg', '.txt')), 'w')
        lbl_file.close()

        xview_bkg_img_file.write('%s\n' % os.path.join(save_bkg_img_dir, bkg_img_names[idx]))
        xview_bkg_lbl_file.write('%s\n' % os.path.join(save_bkg_lbl_dir, bkg_img_names[idx].replace('.jpg', '.txt')))
    xview_bkg_img_file.close()
    xview_bkg_lbl_file.close()


def get_test_xview_bkg_and_lbl_files():
    args = get_args()
    src_all_dir = args.images_save_dir.split('_{}cls'.format(args.class_num)) [0]
    all_images = glob.glob(os.path.join(src_all_dir, '*.jpg'))
    all_img_names = [os.path.basename(f) for f in all_images]

    airc_dir = args.images_save_dir
    airc_images = glob.glob(os.path.join(airc_dir, '*.jpg'))
    airc_img_names = [os.path.basename(f) for f in airc_images]
    airc_img_prefixes = [x.split('_')[0] for x in airc_img_names]
    airc_img_prefixes = list(set(airc_img_prefixes))

    bkg_img_names = [n for n in all_img_names if n.split('_')[0] not in airc_img_prefixes]

    np.random.seed(args.seed)
    bkg_indices = np.random.permutation(len(bkg_img_names))

    save_bkg_img_dir = src_all_dir + '_xview_bkg/'
    if not os.path.exists(save_bkg_img_dir):
        os.mkdir(save_bkg_img_dir)

    list_dir = os.path.join(args.data_save_dir, 'xview_bkg_only_seed{}/'.format(args.seed))
    if not os.path.exists(list_dir):
        os.mkdir(list_dir)
    xview_bkg_img_file = open(os.path.join(list_dir, 'xview_bkg_img_seed{}.txt'.format(args.seed)), 'w')
    xview_bkg_lbl_file = open(os.path.join(list_dir, 'xview_bkg_lbl_seed{}.txt'.format(args.seed)), 'w')

    save_bkg_lbl_dir = args.annos_save_dir[:-1] + '_bkg/'
    if not os.path.exists(save_bkg_lbl_dir):
        os.mkdir(save_bkg_lbl_dir)

    num_airc_imgs = len(airc_img_names)
    for i in range(num_airc_imgs):
        idx = bkg_indices[i]
        shutil.copy(os.path.join(src_all_dir, bkg_img_names[idx]),
                    os.path.join(save_bkg_img_dir, bkg_img_names[idx]))
        lbl_file = open(os.path.join(save_bkg_lbl_dir, bkg_img_names[idx].replace('.jpg', '.txt')), 'w')
        lbl_file.close()

        xview_bkg_img_file.write('%s\n' % os.path.join(save_bkg_img_dir, bkg_img_names[idx]))
        xview_bkg_lbl_file.write('%s\n' % os.path.join(save_bkg_lbl_dir, bkg_img_names[idx].replace('.jpg', '.txt')))
    xview_bkg_img_file.close()
    xview_bkg_lbl_file.close()

def get_args(px_thres=None, whr=None): #
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str,
                        help="Path to folder containing image chips (ie 'Image_Chips/') ",
                        default='/media/lab/Yang/data/xView/train_images/')
    parser.add_argument("--base_tif_folder", type=str,
                        help="Path to folder containing tifs ",
                        default='/media/lab/Yang/data/xView/')
    parser.add_argument("--images_save_dir", type=str, help="to save chip trn val images files",
                        default='/media/lab/Yang/data/xView_YOLO/images/')

    parser.add_argument("--json_filepath", type=str, help="Filepath to GEOJSON coordinate file",
                        default='/media/lab/Yang/data/xView/xView_train.geojson')

    parser.add_argument("--xview_yolo_dir", type=str, help="dir to xViewYOLO",
                        default='/media/lab/Yang/data/xView_YOLO/')

    parser.add_argument("--txt_save_dir", type=str, help="to save  related label files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')
    parser.add_argument("--data_list_save_dir", type=str, help="to save selected trn val images and labels",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_cls/data_list/')

    parser.add_argument("--annos_save_dir", type=str, help="to save txt annotation files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')

    parser.add_argument("--fig_save_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/figures/')

    parser.add_argument("--data_save_dir", type=str, help="to save data files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_cls/')

    parser.add_argument("--cat_sample_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/')

    parser.add_argument("--syn_data_dir", type=str, help="synthetic images",
                        default='/media/lab/Yang/data/synthetic_data/')

    parser.add_argument("--class_num", type=int, default=1, help="Number of Total Categories")  # 60  6
    parser.add_argument("--input_size", type=int, default=608, help="Number of Total Categories")
    parser.add_argument("--seed", type=int, default=17, help="random seed")

    args = parser.parse_args()
    args.images_save_dir = args.images_save_dir + '{}_{}cls/'.format(args.input_size, args.class_num)
    if px_thres:
        args.annos_save_dir = args.annos_save_dir + '{}/{}_cls_xcycwh_px{}whr{}/'.format(args.input_size, args.class_num, px_thres, whr_thres)
    else:
        args.annos_save_dir = args.annos_save_dir + '{}/{}_cls_xcycwh/'.format(args.input_size, args.class_num)
    args.txt_save_dir = args.txt_save_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)
    args.cat_sample_dir = args.cat_sample_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)
    args.data_save_dir = args.data_save_dir.format(args.class_num)
    args.data_list_save_dir = args.data_list_save_dir.format(args.input_size, args.class_num)

    if not os.path.exists(args.txt_save_dir):
        os.makedirs(args.txt_save_dir)

    if not os.path.exists(args.annos_save_dir):
        os.makedirs(args.annos_save_dir)

    if not os.path.exists(args.images_save_dir):
        os.makedirs(args.images_save_dir)

    if not os.path.exists(args.cat_sample_dir):
        os.makedirs(args.cat_sample_dir)

    if not os.path.exists(args.data_save_dir):
        os.makedirs(args.data_save_dir)
    if not os.path.exists(args.data_list_save_dir):
        os.makedirs(args.data_list_save_dir)
    return args


if __name__ == '__main__':
    args = get_args()

    '''
    get images of xview background that not contian aircrafts
    create xivew background label files
    create xview bkg images and labels list files
    '''
    # get_xview_bkg_and_lbl_files_create_realated_list_files()

    '''
    create list of test xview background
    '''


    '''
    remove bad images according to  /media/lab/Yang/data/xView/sailboat_bad_raw_tif_names.txt
    983.tif
    '''
    # bad_img_path = '/media/lab/Yang/data/xView/sailboat_bad_raw_tif_names.txt'
    # src_dir = os.path.join(args.base_tif_folder, 'raw_1_tifs')
    # pwv.remove_bad_image(bad_img_path, src_dir)

    # bad_img_path = '/media/lab/Yang/data/xView/sailboat_bad_image_names.txt'
    # src_dir = args.images_save_dir
    # pwv.remove_bad_image(bad_img_path, src_dir)

    # bad_img_path = '/media/lab/Yang/data/xView/airplane_bad_raw_tif_names.txt'
    # src_dir = os.path.join(args.base_tif_folder, 'raw_0_tifs')
    # pwv.remove_bad_image(bad_img_path, src_dir)

    '''
    backup ground truth *.txt 
    remove bbox and annotations of bad cropped .jpg 
    manually get ***** /media/lab/Yang/data/xView/sailboat_airplane_removed_cropped_jpg_names.txt
    '''
    # bad_img_names = '/media/lab/Yang/data/xView/sailboat_airplane_removed_cropped_jpg_names.txt'
    # args = get_args(px_thres=23, whr=3)
    # pwv.remove_txt_and_json_of_bad_image(bad_img_names, args)



    # whr_thres = 3 # 3.5
    # px_thres= 23
    # args = get_args(px_thres, whr_thres)
    # save_path = args.cat_sample_dir + 'image_with_bbox/'
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # img_list = np.sort(glob.glob(os.path.join(args.images_save_dir, '*.jpg')))
    # for img in img_list:
    #     lbl_name = os.path.basename(img).replace('.jpg', '.txt')
    #     lbl_file = os.path.join(args.annos_save_dir, lbl_name)
    #     gbc.plot_img_with_bbx(img, lbl_file, save_path, label_index=False)
