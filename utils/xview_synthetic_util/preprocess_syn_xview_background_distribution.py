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
from utils.object_score_util import get_bbox_coords_from_annos_with_object_score as gbc

IMG_FORMAT = '.png'
TXT_FORMAT = '.txt'


def merge_clean_origin_syn_image_files(seedians, dt):
    '''
    merge all the origin synthetic data into one folder
    then remove rgb images those contain more than white_thresh*100% white pixels
    and remove gt images that are all white pixels
    :return:
    '''
    step = syn_args.tile_size * syn_args.resolution
    image_folder_name = 'syn_xview_background_images_{}_step{}_sd{}'
    label_folder_name = 'syn_xview_background_annos_{}_step{}_sd{}'

    file_path = syn_args.syn_data_dir.format(dt)

    new_img_folder = '{}_all_images_step{}'.format(dt, step)
    new_lbl_folder = '{}_all_annos_step{}'.format(dt, step)
    des_img_path = os.path.join(file_path, new_img_folder)
    des_lbl_path = os.path.join(file_path, new_lbl_folder)
    if not os.path.exists(des_img_path):
        os.mkdir(des_img_path)
    else:
        shutil.rmtree(des_img_path)
        os.mkdir(des_img_path)
    if not os.path.exists(des_lbl_path):
        os.mkdir(des_lbl_path)
    else:
        shutil.rmtree(des_lbl_path)
        os.mkdir(des_lbl_path)

    for sd in seedians:
        image_path = os.path.join(file_path, image_folder_name.format(dt, step, sd))
        image_files = np.sort(glob.glob(os.path.join(image_path, '*{}'.format(IMG_FORMAT))))
        for img in image_files:
            shutil.copy(img, des_img_path)

        lbl_path = os.path.join(file_path, label_folder_name.format(dt, step, sd))
        lbl_files = np.sort(glob.glob(os.path.join(lbl_path, '*{}'.format(IMG_FORMAT))))

        for lbl in lbl_files:
            shutil.copy(lbl, des_lbl_path)


def group_object_annotation_and_draw_bbox(dt):
    '''
    group annotation files, generate bbox for each object,

    and draw bbox for each ground truth files
    '''
    step = syn_args.tile_size * syn_args.resolution
    folder_name = '{}_all_annos_step{}'.format(dt, step)
    file_path = syn_args.syn_data_dir.format(dt)
    lbl_path = os.path.join(file_path, folder_name)
    txt_folder_name = 'minr{}_linkr{}_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r,
                                                                      dt, step)
    save_txt_path = os.path.join(syn_args.syn_annos_dir, txt_folder_name)
    if not os.path.exists(save_txt_path):
        os.makedirs(save_txt_path)
    else:
        shutil.rmtree(save_txt_path)
        os.makedirs(save_txt_path)

    gbc.get_object_bbox_after_group(lbl_path, save_txt_path, class_label=0, min_region=syn_args.min_region,
                                    link_r=syn_args.link_r)
    gt_files = np.sort(glob.glob(os.path.join(lbl_path, '*{}'.format(IMG_FORMAT))))
    bbox_folder_name = 'minr{}_linkr{}_{}_all_annos_with_bbox_step{}'.format(syn_args.min_region, syn_args.link_r,
                                                                             dt, step)
    save_bbx_path = os.path.join(syn_args.syn_txt_dir, bbox_folder_name)
    if not os.path.exists(save_bbx_path):
        os.makedirs(save_bbx_path)
    else:
        shutil.rmtree(save_bbx_path)
        os.makedirs(save_bbx_path)
    for g in gt_files:
        gt_name = g.split('/')[-1]
        txt_name = gt_name.replace(IMG_FORMAT, TXT_FORMAT)
        txt_file = os.path.join(save_txt_path, txt_name)
        gbc.plot_img_with_bbx(g, txt_file, save_bbx_path)


def draw_bbx_on_rgb_images(dt):
    step = syn_args.tile_size * syn_args.resolution
    img_folder_name = '{}_all_images_step{}'.format(dt, step)
    img_path = os.path.join(syn_args.syn_data_dir.format(dt), img_folder_name)
    img_files = np.sort(glob.glob(os.path.join(img_path, '*{}'.format(IMG_FORMAT))))
    img_names = [os.path.basename(f) for f in img_files]

    txt_folder_name = 'minr{}_linkr{}_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r,
                                                                      dt, step)
    annos_path = os.path.join(syn_args.syn_annos_dir, txt_folder_name)

    bbox_folder_name = 'minr{}_linkr{}_{}_all_images_with_bbox_step{}'.format(syn_args.min_region, syn_args.link_r,
                                                                              dt, step)
    save_bbx_path = os.path.join(syn_args.syn_txt_dir, bbox_folder_name)
    if not os.path.exists(save_bbx_path):
        os.makedirs(save_bbx_path)
    else:
        shutil.rmtree(save_bbx_path)
        os.makedirs(save_bbx_path)

    for ix, f in enumerate(img_files):
        txt_file = os.path.join(annos_path, img_names[ix].replace(IMG_FORMAT, TXT_FORMAT))
        gbc.plot_img_with_bbx(f, txt_file, save_bbx_path, label_index=False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str,
                        help="Path to folder containing image chips (ie 'Image_Chips/') ",
                        default='/media/lab/Yang/data/xView/train_images/')
    parser.add_argument("--raw_folder", type=str,
                        help="Path to folder containing raw images ",
                        default='/media/lab/Yang/data/xView/')

    parser.add_argument("--syn_data_dir", type=str,
                        help="Path to folder containing synthetic images and annos ",
                        default='/media/lab/Yang/data/synthetic_data/syn_xview_background_{}/')
    parser.add_argument("--syn_annos_dir", type=str, default='/media/lab/Yang/data/synthetic_data/syn_xview_background_txt_xcycwh',
                        help="syn xview txt")
    parser.add_argument("--syn_txt_dir", type=str, default='/media/lab/Yang/data/synthetic_data/syn_xview_background_gt_bbox',
                        help="syn xview txt related files")

    parser.add_argument("--syn_display_type", type=str, default='color',
                        help="texture, color, mixed")  # syn_color0, syn_texture0,

    parser.add_argument("--min_region", type=int, default=100, help="the smallest #pixels (area) to form an object")
    parser.add_argument("--link_r", type=int, default=15,
                        help="the #pixels between two connected components to be grouped")
    parser.add_argument("--resolution", type=float, default=0.3, help="resolution of synthetic data")
    parser.add_argument("--tile_size", type=int, default=608, help="image size")

    args = parser.parse_args()
    # if not os.path.exists(args.syn_annos_dir):
    #     os.makedirs(args.syn_annos_dir)
    # if not os.path.exists(args.syn_txt_dir):
    #     os.makedirs(args.syn_txt_dir)
    return args


if __name__ == '__main__':
    syn_args = get_args()
    '''
    get raw images contain more than 30% black pixels
    get tif those contain airplanes
    get no airplane raw images
    get shape raw images
    '''
    # get_black_raw_img_list()

    # get_tif_contain_airplanes()

    # get_no_airplane_raw_images()

    # get_shape_raw_images()

    '''
    merge all syn_xveiw_background data
    '''
    # seedians = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # display_types = ['texture', 'color', 'mixed']
    # for dt in display_types:
    #     merge_clean_origin_syn_image_files(seedians, dt)

    '''
    generate txt and bbox for syn_xveiw_background data
    bbox annotation meet certain conditions: px_thres, whr_thres
    '''
    # display_types = ['texture', 'color', 'mixed']
    # for dt in display_types:
    #     group_object_annotation_and_draw_bbox(dt)

    '''
    draw bbox on rgb images for syn_xveiw_background data
    '''
    # display_types = ['texture', 'color', 'mixed']
    # for dt in display_types:
    #     draw_bbx_on_rgb_images(dt)

