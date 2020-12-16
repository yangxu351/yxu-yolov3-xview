'''
xview process
in order to generate xivew 2-d background with synthetic airplances
'''

import sys
sys.path.append('/data/users/yang/code/yxu-yolov3-xview/')
import glob
import numpy as np
import argparse
import os
from PIL import Image
import pandas as pd
import shutil
from utils.object_score_util import get_bbox_coords_from_annos_with_object_score as gbc
from skimage import io
import matplotlib.pyplot as plt
import cv2

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


def resize_crop(cmt, display_types, scale=2):
    for dt in display_types:
        sys_args = get_args(cmt)
        step = sys_args.resolution * sys_args.tile_size
        save_img_dir = os.path.join(sys_args.syn_data_dir.format(dt), 'upscale_{}_all_images_step{}'.format(dt, step))
        if not os.path.exists(save_img_dir):
            os.mkdir(save_img_dir)
        save_lbl_dir = os.path.join(sys_args.syn_data_dir.format(dt), 'upscale_{}_all_annos_step{}'.format(dt, step))
        if not os.path.exists(save_lbl_dir):
            os.mkdir(save_lbl_dir)
        img_path = os.path.join(sys_args.syn_data_dir.format(dt), '{}_all_images_step{}'.format(dt, step))
        lbl_path = os.path.join(sys_args.syn_data_dir.format(dt), '{}_all_annos_step{}'.format(dt, step))
        img_list = np.sort(glob.glob(os.path.join(img_path, '*.png')))
        for f in img_list:
            img = cv2.imread(f)
            h, w, _ = img.shape
            img2 = cv2.resize(img, (h*scale, w*scale), interpolation=cv2.INTER_LINEAR)
            name = os.path.basename(f)
            lbl = cv2.imread(os.path.join(lbl_path, name))
            lbl2 = cv2.resize(lbl, (h*scale, w*scale), interpolation=cv2.INTER_LINEAR)
            for i in range(scale):
                for j in range(scale):
                    img_s = img2[i*w: (i+1)*w, j*w: (j+1)*w]
                    cv2.imwrite(os.path.join(save_img_dir, name.split('.')[0] + '_i{}j{}.png'.format(i, j)), img_s)
                    lbl_s = lbl2[i*w: (i+1)*w, j*w: (j+1)*w]
                    cv2.imwrite(os.path.join(save_lbl_dir, name.split('.')[0] + '_i{}j{}.png'.format(i, j)), lbl_s)



def group_object_annotation_and_draw_bbox(dt, px_thresh=20, whr_thres=4, upscale=False):
    '''
    group annotation files, generate bbox for each object,

    and draw bbox for each ground truth files
    '''
    step = syn_args.tile_size * syn_args.resolution
    if upscale:
        folder_name = 'upscale_{}_all_annos_step{}'.format(dt, step)
        txt_folder_name = 'minr{}_linkr{}_px{}whr{}_upscale_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, px_thresh, whr_thres,
                                                                      dt, step)
        bbox_folder_name = 'minr{}_linkr{}_px{}whr{}_upscale_{}_all_annos_with_bbox_step{}'.format(syn_args.min_region, syn_args.link_r,
                                                                                       px_thresh, whr_thres, dt, step)
    else:
        folder_name = '{}_all_annos_step{}'.format(dt, step)
        txt_folder_name = 'minr{}_linkr{}_px{}whr{}_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, px_thresh, whr_thres,
                                                                      dt, step)
        bbox_folder_name = 'minr{}_linkr{}_px{}whr{}_{}_all_annos_with_bbox_step{}'.format(syn_args.min_region, syn_args.link_r,
                                                                                       px_thresh, whr_thres, dt, step)
    lbl_path = os.path.join(syn_args.syn_data_dir, folder_name)
    
    save_txt_path = os.path.join(syn_args.syn_annos_dir, txt_folder_name)
    if not os.path.exists(save_txt_path):
        os.makedirs(save_txt_path)
    else:
        shutil.rmtree(save_txt_path)
        os.makedirs(save_txt_path)
    gt_files = np.sort(glob.glob(os.path.join(lbl_path, '*.png')))


    gbc.get_object_bbox_after_group(lbl_path, save_txt_path, class_label=0, min_region=syn_args.min_region, link_r=syn_args.link_r, px_thresh=px_thresh, whr_thres=whr_thres)
    gt_files = np.sort(glob.glob(os.path.join(lbl_path, '*.png')))
    print('gt_files', len(gt_files))
    save_bbx_path = os.path.join(syn_args.syn_txt_dir, bbox_folder_name)
    if not os.path.exists(save_bbx_path):
        os.makedirs(save_bbx_path)
    else:
        shutil.rmtree(save_bbx_path)
        os.makedirs(save_bbx_path)
#    for g in gt_files:
#        gt_name = g.split('/')[-1]
#        txt_name = gt_name.replace('.png', '.txt')
#        txt_file = os.path.join(save_txt_path, txt_name)
#        gbc.plot_img_with_bbx(g, txt_file, save_bbx_path)


def draw_bbx_on_rgb_images(dt, px_thresh=20, whr_thres=4):
    step = syn_args.tile_size * syn_args.resolution
    img_folder_name = '{}_all_images_step{}'.format(dt, step)
    img_path = os.path.join(syn_args.syn_data_dir, img_folder_name)
    img_files = np.sort(glob.glob(os.path.join(img_path, '*.png')))
    img_names = [os.path.basename(f) for f in img_files]

    txt_folder_name = 'minr{}_linkr{}_px{}whr{}_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, px_thresh, whr_thres,
                                                                                dt, step)
    annos_path = os.path.join(syn_args.syn_annos_dir, txt_folder_name)

    bbox_folder_name = 'minr{}_linkr{}_px{}whr{}_{}_all_images_with_bbox_step{}'.format(syn_args.min_region, syn_args.link_r, px_thresh, whr_thres,
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


def plot_rgb_histogram(img_path, syn=False):
    save_dir = '/data/users/yang/data/xView_YOLO/cat_samples/608/1_cls/rgb_histogram/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if syn:
        x_files = glob.glob(os.path.join(img_path, '*.png'))
        name = img_path.split('/')[-3] + '.png'
        title = '{} RGB Histogram'.format(img_path.split('/')[-3])
    else:
        x_files = glob.glob(os.path.join(img_path, '*.jpg'))
        name = 'xview_rgb_histogram.jpg'
        title = 'Xview RGB Histogram'
    arr_hist = np.zeros(shape=(3, 256, len(x_files)))
    color = ('b','g','r')
    for ix, f in enumerate(x_files):
        img = cv2.imread(f)
        for i,col in enumerate(color):
            # (256, 1)
            histr = cv2.calcHist([img],[i],None,[256],[0,256])
            arr_hist[i, :, ix] = histr[:, 0]

    hs = arr_hist.mean(axis=-1)
    print(hs.shape)
    for cx, c in enumerate(color):
        plt.plot(hs[cx, :],color = c)

    plt.xlim([0,256])
    plt.xlabel('Intensity Value')
    plt.ylabel('Count')
    plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])
    plt.suptitle(title)
    plt.savefig(os.path.join(save_dir, name))
    plt.show()


def get_args(cmt, CC=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str,
                        help="Path to folder containing image chips (ie 'Image_Chips/') ",
                        default='/data/users/yang/data/xView/train_images/')
    parser.add_argument("--raw_folder", type=str,
                        help="Path to folder containing raw images ",
                        default='/data/users/yang/data/xView/')

    #fixme
    if CC: # certain_models, scale_models
        parser.add_argument("--syn_data_dir", type=str,
                        help="Path to folder containing synthetic images and annos ",
                        default='/data/users/yang/data/synthetic_data_CC/syn_xview_bkg_{}'.format(cmt))
        parser.add_argument("--syn_annos_dir", type=str, default='/data/users/yang/data/synthetic_data_CC/syn_xview_bkg_{}_txt_xcycwh/'.format(cmt),
                            help="syn xview txt")
        parser.add_argument("--syn_txt_dir", type=str, default='/data/users/yang/data/synthetic_data_CC/syn_xview_bkg_{}_gt_bbox/'.format(cmt),
                            help="syn xview txt related files")
    else: # cmt == ''
        parser.add_argument("--syn_data_dir", type=str,
                        help="Path to folder containing synthetic images and annos ",
                        default='/data/users/yang/data/synthetic_data/syn_xview_bkg_{}'.format(cmt))
        parser.add_argument("--syn_annos_dir", type=str, default='/data/users/yang/data/synthetic_data/syn_xview_bkg_{}_txt_xcycwh/'.format(cmt),
                            help="syn xview txt")
        parser.add_argument("--syn_txt_dir", type=str, default='/data/users/yang/data/synthetic_data/syn_xview_bkg_{}_gt_bbox/'.format(cmt),
                            help="syn xview txt related files")

    parser.add_argument("--syn_display_type", type=str, default='color',
                        help="texture, color, mixed")  # syn_color0, syn_texture0,
    #fixme ---***** min_region ***** change
    parser.add_argument("--min_region", type=int, default=100, help="300 100 the smallest #pixels (area) to form an object")
    parser.add_argument("--link_r", type=int, default=15,
                        help="the #pixels between two connected components to be grouped")
    parser.add_argument("--resolution", type=float, default=0.3, help="resolution of synthetic data")
    parser.add_argument("--tile_size", type=int, default=608, help="image size")
    parser.add_argument("--class_num", type=int, default=1, help="class number")

    args = parser.parse_args()
    # if not os.path.exists(args.syn_annos_dir):
    #     os.makedirs(args.syn_annos_dir)
    # if not os.path.exists(args.syn_txt_dir):
    #     os.makedirs(args.syn_txt_dir)

    return args


if __name__ == '__main__':

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
    *****---change syn_data_dir first----******
    '''
    # cmt = ''
    # seedians = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # display_types = ['texture', 'color', 'mixed']
    # cmt = 'certain_models'
    # display_types = ['texture', 'color', 'mixed']
    # seedians = [0, 1, 2, 3, 4, 5, 6, 7]
    # cmt = 'scale_models'
    # display_types = ['color', 'mixed']
    # seedians = [0, 1, 2, 3, 4, 5, 6, 7]
    # cmt = 'small_models'
    # # # cmt = 'small_fw_models'
    # cmt = '6groups_models'
    # cmt = 'rnd_bwratio_models'
    # cmt = 'rnd_bwratio_flat0.8_models'
    # cmt = 'rnd_bwratio_asx_models'
    # cmt = 'xratio_xcolor_models'
    # cmt = 'sbwratio_xratio_xcolor_models'
    # cmt = 'sbwratio_xratio_xcolor_dark_models'
    # cmt = 'sbwratio_new_xratio_xcolor_models'
    # display_types = ['color', 'mixed']
    # seedians = [0, 1, 2, 3, 4]
    # syn_args = get_args(cmt)
    # for dt in display_types:
    #     merge_clean_origin_syn_image_files(seedians, dt)

    '''
    draw RGB histogram 
    '''
    # xview_patch_dir = '/data/users/yang/data/xView_YOLO/images/608_1cls/'
    # plot_rgb_histogram(xview_patch_dir, syn=False)

    # syn_patch_dir = '/data/users/yang/data/synthetic_data/syn_xview_bkg_sbwratio_xratio_xcolor_models_color/color_all_images_step182.4/'
    # syn_patch_dir = '/data/users/yang/data/synthetic_data/syn_xview_bkg_sbwratio_xratio_xcolor_models_mixed/mixed_all_images_step182.4/'
    # syn_patch_dir = '/data/users/yang/data/synthetic_data/syn_xview_bkg_sbwratio_xratio_xcolor_dark_models_color/color_all_images_step182.4/'
    # syn_patch_dir = '/data/users/yang/data/synthetic_data/syn_xview_bkg_sbwratio_xratio_xcolor_dark_models_mixed/mixed_all_images_step182.4/'
    # syn_patch_dir = '/data/users/yang/data/synthetic_data/Airplanes/syn_color/syn_color_all_images_step182.4/'
    # syn_patch_dir = '/data/users/yang/data/synthetic_data/syn_xview_bkg_sbwratio_new_xratio_xcolor_models_mixed/mixed_all_images_step182.4/'
    # syn_patch_dir = '/data/users/yang/data/synthetic_data/syn_xview_bkg_sbwratio_new_xratio_xcolor_models_mixed/mixed_all_images_step182.4/'
    # plot_rgb_histogram(syn_patch_dir, syn=True)

    '''
    resize images and labels 
    2x upsample
    crop 
    '''
    # cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_gauss_50_model4_v20'
#        cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_gauss_120_model4_v21'
#    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_uniform_50_model4_v19'
#    cmt = 'xbw_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_rnd_color_bias0_model4_v21'
#    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_bias0_model4_v1'
#    cmt = 'xbw_xcolor_xbkg_unif_shdw_scatter_gauss_50_color_bias0_model5_RC5_v1'
#    display_types = ['color'] # , 'mixed']
#    resize_crop(cmt, display_types, scale=2)

    '''
    generate txt and bbox for syn_xveiw_background data
    bbox annotation meet certain conditions: px_thres, whr_thres
    '''

#    size_sigma = [0, 0.03, 0.06, 0.09, 0.12] # 0, 0.03, 0.06, 0.09, 0.12
#    for ix,ssig in enumerate(size_sigma):
##        cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_promu_size_bias{}_RC1_v{}'.format(ssig, ix+40) 
##        cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias{}_RC1_v{}'.format(ssig, ix+100) 
##        px_thres = 15
##        cmt = 'xbw_newbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias{}_RC1_v{}'.format(ssig, ix+110) 
##        px_thres = 15
#        
##        cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_promu_size_bias{}_RC2_v{}'.format(ssig, ix+40)
##        px_thres = 23
##        cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias{}_RC2_v{}'.format(ssig, ix+100)
##        px_thres = 23
##        cmt = 'xbsw_xwing_newbkg_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias{}_RC2_v{}'.format(ssig, ix+110)
##        px_thres = 23
#
##        cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_size_bias{}_RC3_v{}'.format(ssig, ix+40)
##        px_thres = 23
##        cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias{}_RC3_v{}'.format(ssig, ix+100)
##        px_thres = 23
##        cmt = 'xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias{}_RC3_v{}'.format(ssig, ix+110)
##        px_thres = 23
#
##        cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_size_bias{}_RC4_v{}'.format(ssig, ix+40)
##        px_thres = 23
##        cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias{}_RC4_v{}'.format(ssig, ix+100)
##        px_thres = 23
##        cmt = 'xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias{}_RC4_v{}'.format(ssig, ix+110)
##        px_thres = 23
#
##        cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_size_bias{}_RC5_v{}'.format(ssig, ix+40)
##        px_thres = 23
##        cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias{}_RC5_v{}'.format(ssig, ix+100)
##        px_thres = 23
#        cmt = 'xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias{}_RC5_v{}'.format(ssig, ix+110)
#        px_thres = 23

#    color_sigma = [5, 10, 15, 20] # 0, 
#    for ix,ssig in enumerate(color_sigma):
#        cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_ssig0_color_bias{}_RC1_v{}'.format(ssig, ix+50)
#        px_thres = 15 
##        cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_ssig0.03_color_bias{}_RC2_v{}'.format(ssig, ix+31)
##        px_thres = 23
##        cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0_color_bias{}_RC3_v{}'.format(ssig, ix+11)
##        px_thres = 23
#        cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.06_color_bias{}_RC4_v{}'.format(ssig, ix+31)
#        cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_square_bias{}_RC4_v{}'.format(ssig, ix+71)
#        px_thres = 23
##        cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.03_color_bias{}_RC5_v{}'.format(ssig, ix+31)
##        px_thres = 23

#    color_sigma = [0] # 0, 15, 30, 45, 60
#    for ix,ssig in enumerate(color_sigma):
#        cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_ssig0_color_bias{}_RC1_v{}'.format(ssig, ix+51)
#        px_thres = 15 
#        cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_ssig0_color_bias{}_RC2_v{}'.format(ssig, ix+50)
#        px_thres = 23
#        cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_bias{}_RC3_v{}'.format(ssig, ix+50)
#        px_thres = 23
#        cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_bias{}_RC4_v{}'.format(ssig, ix+50)
#        px_thres = 23
#        cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.03_color_bias{}_RC5_v{}'.format(ssig, ix+50)
#        px_thres = 23
    
#    color_sigma = [10, 20, 30, 40] # 10, 20, 30, 40
#    #color_sigma = [0]
#    base_version = 121
#    for ix, csig in enumerate(color_sigma):
##        #cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_ssig0_color_square_bias{}_RC1_v{}'.format(csig, ix+91)
##        #cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_ssig0.06_color_square_bias{}_RC1_v{}'.format(csig, ix+96)
###        cmt = 'xbw_newbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_ssig0.03_color_square_bias{}_RC1_v{}'.format(csig, ix+116)
###        px_thres = 15 
##
##        cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_ssig0_color_square_bias{}_RC2_v{}'.format(csig, ix+91)
##        cmt = 'xbsw_xwing_newbkg_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_square_bias{}_RC2_v{}'.format(csig, ix+116)
##        px_thres = 23
##        cmt = 'xbsw_xwing_newbkg_shdw_split_scatter_gauss_rndsolar_ssig0.03_color_square_bias{}_RC2_v{}'.format(csig, ix+base_version)
##        px_thres = 23
#        
###        cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_square_bias{}_RC3_v{}'.format(csig, ix+91)
##        cmt = 'xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.03_color_square_bias{}_RC3_v{}'.format(csig, ix+116)
##        px_thres = 23
#        cmt = 'xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_square_bias{}_RC3_v{}'.format(csig, ix+base_version)
#        px_thres = 23
###
###        cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_square_bias{}_RC4_v{}'.format(csig, ix+91)
##        cmt = 'xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.12_color_square_bias{}_RC4_v{}'.format(csig, ix+116)
##        px_thres = 23
###        
##        cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0_color_square_bias{}_RC5_v{}'.format(csig, ix+91)
##        cmt = 'xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.03_color_square_bias{}_RC5_v{}'.format(csig, ix+116)
##        px_thres = 23
#        
#        whr_thres = 3
#        dt = 'color'
#        syn_args = get_args(cmt)
#        group_object_annotation_and_draw_bbox(dt, px_thres, whr_thres)
#        draw_bbx_on_rgb_images(dt, px_thres, whr_thres)
        
#    ''' optimize sigma size for common class '''    
#    size_sigma = []#0, 0.08, 0.16, 0.24, 0.32] # for CC1 
##    size_sigma = [0, 0.06, 0.12, 0.18, 0.24]# for CC2 0, 0.06, 0.12, 0.18, 0.24
#    for ix,ssig in enumerate(size_sigma):
##        cmt = 'unif_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias{}_CC1_v{}'.format(ssig, ix+10)
##        px_thres = 23
##        cmt = 'xbsw_xwing_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias{}_CC2_v{}'.format(ssig, ix+15)
##        px_thres = 23
#
##        cmt = 'unif_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias{}_CC1_v{}'.format(ssig, ix+20)
##        px_thres = 23
##        cmt = 'shdw_split_scatter_gauss_rndsolar_promu_size_square_bias{}_CC2_v{}'.format(ssig, ix+20)
##        px_thres = 23
# 
##        cmt = 'unif_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias{}_CC1_v{}'.format(ssig, ix+20)
#        cmt = 'new_bkg_unif_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias{}_CC1_v{}'.format(ssig, ix+40)
#        px_thres = 23
#
##        cmt = 'shdw_split_scatter_gauss_rndsolar_promu_size_square_bias{}_CC2_v{}'.format(ssig, ix+20)
##        cmt = 'new_bkg_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias{}_CC2_v{}'.format(ssig, ix+40)
##        px_thres = 23
#       
#        whr_thres = 3
#        dt = 'color'
#        syn_args = get_args(cmt, CC=True)
#        group_object_annotation_and_draw_bbox(dt, px_thres, whr_thres)
#        draw_bbx_on_rgb_images(dt, px_thres, whr_thres)

    ''' optimize sigma color for common class '''    
#    color_sigma = [1, 2, 3, 4]#1, 2, 3
##    color_sigma = [0, 1, 2, 3, 4]#
#    for ix,csig in enumerate(color_sigma):
##        cmt = 'unif_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias{}_CC1_v{}'.format(ssig, ix+10)
##        px_thres = 23
##        cmt = 'xbsw_xwing_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias{}_CC2_v{}'.format(ssig, ix+15)
##        px_thres = 23
#
##        cmt = 'unif_shdw_split_scatter_gauss_rndsolar_ssig0.16_color_square_bias{}_CC1_v{}'.format(csig, ix+31)
##        px_thres = 23
##        cmt = 'new_bkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.24_color_square_bias{}_CC1_v{}'.format(csig, ix+46)
##        px_thres = 23
##        cmt = 'new_bkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.08_color_square_bias{}_CC1_v{}'.format(csig, ix+51)
##        px_thres = 23
##        cmt = 'shdw_split_scatter_gauss_rndsolar_ssig0.24_color_square_bias{}_CC2_v{}'.format(csig, ix+31)
##        cmt = 'shdw_split_scatter_gauss_rndsolar_ssig0.24_same_bwcolor_square_bias{}_CC2_v{}'.format(csig, ix+36)
##        cmt = 'new_bkg_shdw_split_scatter_gauss_rndsolar_ssig0.12_color_square_bias{}_CC2_v{}'.format(csig, ix+46)
##        cmt = 'new_bkg_shdw_split_scatter_gauss_rndsolar_ssig0.12_new_color_square_bias{}_CC2_v{}'.format(csig, ix+56)
##        
#        cmt = 'new_bkg_shdw_split_scatter_gauss_rndsolar_ssig0_color_square_bias{}_CC2_v{}'.format(csig, ix+51)
#        px_thres = 23
#        
#        whr_thres = 3
#        dt = 'color'
#        syn_args = get_args(cmt, CC=True)
#        group_object_annotation_and_draw_bbox(dt, px_thres, whr_thres)
#        draw_bbx_on_rgb_images(dt, px_thres, whr_thres)
     
    ############# quantities     
#    quantities = ['new_bkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.24_color_square_bias2_CC1_quantities',
#    'new_bkg_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias0.12_CC2_quantities']
#    px_thres = 23
#    for cmt in quantities:
#        whr_thres = 3
#        dt = 'color'
#        syn_args = get_args(cmt, CC=True)
#        group_object_annotation_and_draw_bbox(dt, px_thres, whr_thres)
#        draw_bbx_on_rgb_images(dt, px_thres, whr_thres)      
    
    ############ 1 instance     
#    instance = ['new_bkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0_color_square_bias0_CC1_1inst_v63',
#    'new_bkg_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias0_CC2_1inst_v63']
#    instance = ['new_bkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.24_color_square_bias2_CC1_1inst_v64',
#    'new_bkg_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias0.12_CC2_1inst_v64']
#    instance = ['new_bkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.08_csig1_CC1_1inst_v65',
#    'new_bkg_shdw_split_scatter_gauss_rndsolar_promu_ssig0_csig4_CC2_1inst_v65']
#    px_thres = 23
#    for cmt in instance[:1]:
#        whr_thres = 3
#        dt = 'color'
#        syn_args = get_args(cmt, CC=True)
#        group_object_annotation_and_draw_bbox(dt, px_thres, whr_thres)
#        draw_bbx_on_rgb_images(dt, px_thres, whr_thres)  
    
    #############  increse size color mu   
    mu_list = [10, 20, 30]
    for ix, ml in enumerate(mu_list):
        ########## size
        #cmt = 'xbw_newbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_promu_ssig0.03_increase_size_mu_{}_RC1_v{}'.format(ml, ix+131)
        #cmt = 'xbsw_xwing_newbkg_shdw_split_scatter_gauss_rndsolar_promu_ssig0.03_increase_size_mu_{}_RC2_v{}'.format(ml, ix+131)
        #cmt = 'xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_ssig0.09_increase_size_mu_{}_RC3_v{}'.format(ml, ix+131)
        #cmt = 'xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_ssig0.12_increase_mu_{}_RC4_v{}'.format(ml, ix+121)
        #cmt = 'xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_ssig0.03_increase_mu_{}_RC5_v{}'.format(ml, ix+121)
        ###########  color 
        #cmt = 'xbw_newbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_promu_ssig0.03_increase_color_mu_{}_RC1_v{}'.format(ml, ix+136)
        #cmt = 'xbsw_xwing_newbkg_shdw_split_scatter_gauss_rndsolar_promu_ssig0.03_csig10_increase_color_mu_{}_RC2_v{}'.format(ml, ix+136)
        #cmt = 'xbsw_xwing_newbkg_shdw_split_scatter_gauss_rndsolar_promu_ssig0.03_csig10_decrease_color_mu_{}_RC2_v{}'.format(ml, ix+136)
        #cmt = 'xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_ssig0.09_csig40_increase_color_mu_{}_RC3_v{}'.format(ml, ix+136)
        cmt = 'xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_ssig0.09_csig40_decrease_color_mu_{}_RC3_v{}'.format(ml, ix+136)
        #cmt = 'xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_ssig0.12_increase_color_mu_{}_RC4_v{}'.format(ml, ix+126)
        #cmt = 'xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_ssig0.03_increase_color_mu_{}_RC5_v{}'.format(ml, ix+126)
        
        px_thres = 23
        whr_thres = 3
        dt = 'color'
        syn_args = get_args(cmt, CC=False)
        group_object_annotation_and_draw_bbox(dt, px_thres, whr_thres)
        draw_bbx_on_rgb_images(dt, px_thres, whr_thres)    


    ######## scatter for CC1
    #cmt = 'new_bkg_unif_shdw_scatter_gauss_rndsolar_promu_size_square_bias0_CC1_v70'
#    cmt = 'new_bkg_unif_shdw_scatter_gauss_rndsolar_promu_size_square_bias0_large_CC1_v71' 
#    px_thres = 23
#    whr_thres = 3
#    dt = 'color'
#    syn_args = get_args(cmt, CC=True)
#    group_object_annotation_and_draw_bbox(dt, px_thres, whr_thres)
#    draw_bbx_on_rgb_images(dt, px_thres, whr_thres)   
    
#    cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_ssig0_texture_RC1_v80'
#    px_thres = 15
    
#    cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_ssig0_texture_RC2_v80'
#    px_thres = 23
    
#    cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_texture_RC4_v80'
#    px_thres = 23
#    
#    whr_thres = 3
##        dt = 'color'
#    dt = 'texture'
#    syn_args = get_args(cmt)
#    group_object_annotation_and_draw_bbox(dt, px_thres, whr_thres)
#    draw_bbx_on_rgb_images(dt, px_thres, whr_thres)


    #cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_fixedsolar_ssig0.03_csig20_RC1_v30'
    #px_thres = 15
    #cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_fixedsolar_ssig0.12_csig10_RC2_v30'
    #px_thres = 23
    #cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_fixedsolar_ssig0_csig0_RC3_v30'
    #px_thres = 23
    #cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_fixedsolar_ssig0.09_csig0_RC4_v30'
    #px_thres = 23
    #cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_fixedsolar_ssig0.06_csig10_RC5_v30'
    #px_thres = 23    
    
#    whr_thres = 3
#    dt = 'color'
#    syn_args = get_args(cmt)
#    group_object_annotation_and_draw_bbox(dt, px_thres, whr_thres)
#    draw_bbx_on_rgb_images(dt, px_thres, whr_thres)

#    cmt = 'new_bkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.24_color_square_bias2_CC1_2xdensity_v50'
#    cmt = 'new_bkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.24_color_square_bias2_CC1_4xdensity_v51'
#    cmt = 'new_bkg_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias0.12_CC2_2xdensity_v50'
#    cmt = 'new_bkg_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias0.12_CC2_4xdensity_v51'
#    px_thres = 23
    
#    whr_thres = 3
#    dt = 'color'
#    syn_args = get_args(cmt, CC=True)
#    group_object_annotation_and_draw_bbox(dt, px_thres, whr_thres)
#    draw_bbx_on_rgb_images(dt, px_thres, whr_thres)

