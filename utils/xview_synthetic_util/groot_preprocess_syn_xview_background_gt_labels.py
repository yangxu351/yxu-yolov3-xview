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
    lbl_path = os.path.join(syn_args.syn_data_dir.format(dt), folder_name)
    
    save_txt_path = os.path.join(syn_args.syn_annos_dir, txt_folder_name)
    if not os.path.exists(save_txt_path):
        os.makedirs(save_txt_path)
    else:
        shutil.rmtree(save_txt_path)
        os.makedirs(save_txt_path)

    gbc.get_object_bbox_after_group(lbl_path, save_txt_path, class_label=0, min_region=syn_args.min_region, link_r=syn_args.link_r, px_thresh=px_thresh, whr_thres=whr_thres)
    gt_files = np.sort(glob.glob(os.path.join(lbl_path, '*.png')))
    
    save_bbx_path = os.path.join(syn_args.syn_txt_dir, bbox_folder_name)
    if not os.path.exists(save_bbx_path):
        os.makedirs(save_bbx_path)
    else:
        shutil.rmtree(save_bbx_path)
        os.makedirs(save_bbx_path)
    for g in gt_files:
        gt_name = g.split('/')[-1]
        txt_name = gt_name.replace('.png', '.txt')
        txt_file = os.path.join(save_txt_path, txt_name)
        gbc.plot_img_with_bbx(g, txt_file, save_bbx_path)


def draw_bbx_on_rgb_images(dt, px_thresh=20, whr_thres=4):
    step = syn_args.tile_size * syn_args.resolution
    img_folder_name = '{}_all_images_step{}'.format(dt, step)
    img_path = os.path.join(syn_args.syn_data_dir.format(dt), img_folder_name)
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


def get_args(cmt=''):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str,
                        help="Path to folder containing image chips (ie 'Image_Chips/') ",
                        default='/data/users/yang/data/xView/train_images/')
    parser.add_argument("--raw_folder", type=str,
                        help="Path to folder containing raw images ",
                        default='/data/users/yang/data/xView/')

    #fixme
    if cmt: # certain_models, scale_models
        parser.add_argument("--syn_data_dir", type=str,
                        help="Path to folder containing synthetic images and annos ",
                        default='/data/users/yang/data/synthetic_data/syn_xview_bkg_{}'.format(cmt)+'_{}/')
        parser.add_argument("--syn_annos_dir", type=str, default='/data/users/yang/data/synthetic_data/syn_xview_bkg_{}_txt_xcycwh/'.format(cmt),
                            help="syn xview txt")
        parser.add_argument("--syn_txt_dir", type=str, default='/data/users/yang/data/synthetic_data/syn_xview_bkg_{}_gt_bbox/'.format(cmt),
                            help="syn xview txt related files")
    else: # cmt == ''
        parser.add_argument("--syn_data_dir", type=str,
                            help="Path to folder containing synthetic images and annos ",
                            default='/data/users/yang/data/synthetic_data/syn_xview_background_{}/')
        parser.add_argument("--syn_annos_dir", type=str, default='/data/users/yang/data/synthetic_data/syn_xview_background_txt_xcycwh',
                            help="syn xview txt")
        parser.add_argument("--syn_txt_dir", type=str, default='/data/users/yang/data/synthetic_data/syn_xview_background_gt_bbox',
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
    # # px_thres=6
    # # whr_thres=4
    # # px_thres=20
    # # whr_thres=4
    # px_thres=23 #20 #30
    # whr_thres=4
    # # # display_types = ['texture', 'color', 'mixed']
    # # # # cmt = ''
    # # # # cmt = 'certain_models'
    # # # cmt = 'scale_models'
    # cmt = 'small_models'
    # # cmt = 'small_fw_models'
    # # # cmt = '6groups_models'
    # # cmt = 'rnd_bwratio_models'
    # cmt = 'rnd_bwratio_flat0.8_models'
    # cmt = 'rnd_bwratio_asx_models'
    # cmt = 'xratio_xcolor_models'
    # cmt = 'sbwratio_xratio_xcolor_models'
    # cmt = 'sbwratio_new_xratio_xcolor_models'
    # cmt = 'xbw_xrxc_spr_sml_models'
    # cmt = 'sbw_xcolor_model4'
    # cmt = 'sbw_xcolor_model4_v1'
    # cmt = 'sbw_xcolor_model4_v2'
    # cmt = 'xbw_xcolor_xbkg_gauss_model4_v3'
    # cmt = 'xbw_xcolor_xbkg_gauss_model4_v4'
    # cmt = 'xbw_xcolor_xbkg_unif_mig21_model4_v7'
    #    cmt = 'xbw_xcolor_xbkg_unif_mig21_rndp_model4_v7'
        # cmt = 'xbw_rndcolor_xbkg_unif_mig21_model4_v8'
    #    cmt = 'xbw_rndcolor_xbkg_unif_mig21_rndp_model4_v9'
    #    cmt = 'xbw_xcolor_xbkg_unif_mig21_rndp_shdw_model4_v10'
    #    cmt = 'xbw_rndcolor_xbkg_unif_mig21_rndp_shdw_model4_v11'
    #    cmt = 'xbw_xcolor_xbkg_unif_mig21_rndp_shdw_dense_model4_v12'
    #    cmt = 'xbw_xcolor_xbkg_unif_mig21_rndp_shdw_dense_angle_model4_v13'
    #    cmt = 'xbw_xcolor_xbkg_unif_mig21_rndp_shdw_dense_angle_fwc_model4_v14'
    #    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_angle_fwc_scatter_gauss_dense_model4_v15'
    #    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_angle_fwc_scatter_gauss_dense_model4_v16'
    #    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_gauss_dense_model4_v17'
    #    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_gauss_dense_model4_v17'
    #    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_gauss_dense_solar_model4_v18'
    #    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_uniform_50_model4_v19'
    #    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_gauss_50_model4_v20'
#    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_gauss_120_model4_v21'
#    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_uniform_50_rdegree_model4_v22'
#    cmt = 'xbw_rndcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_model4_v23'
#    cmt = ['xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_bias0_model4_v1', 'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_100_bias0_model4_v2',    'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias20_model4_v3',    'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias40_model4_v4', 'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias50_model4_v5','xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias60_model4_v6', 'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias70_model4_v7', 'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias110_model4_v8', 'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_rnd_model4_v9']

#    px_thres=15
#    px_thres = 30
#        px_thres = 50
    # cmt = 'sbw_xcolor_model0'
    # cmt = 'xbw_xrxc_spr_sml_models_gauss'
    # cmt = 'xbw_xrxc_spr_sml_models_gauss'
    # cmt = 'sbw_xcolor_model1'
    # cmt = 'xbw_xcolor_gauss_model1_v1'
    # cmt = 'xbw_xcolor_xbkg_gauss_model1_v2'
    # cmt = 'sbw_xcolor_xbkg_unif_model1_v3'
    # cmt = 'xbsw_xcolor_xbkg_gauss_model1_v4'
    #    cmt = 'xbsw_xcolor_xbkg_unif_rndp_model1_v6'
    #    cmt = 'xbsw_rndcolor_xbkg_unif_rndp_model1_v7'
    #    cmt = 'xbsw_xwing_color_xbkg_unif_rndp_model1_v8'
    #    cmt = 'xbsw_xwing_color_xbkg_unif_rndp_shdw_model1_v9'
    #    px_thres=23

    #    cmt = 'xbw_xcolor_xbkg_unif_shdw_scatter_uniform_model5_v1'
    #    cmt = 'xbw_xcolor_xbkg_unif_shdw_rndp_model5_v2'
#    cmt = 'xbw_xcolor_xbkg_unif_shdw_scatter_gauss_40_bias0_model5_v1'

#    for color_pro in range(1):
# #       if color_pro == 0:
# #           cmt = 'xbw_xcolor_xbkg_unif_shdw_scatter_gauss_40_bias0_model5_v1'
# #       else:
# #           cmt = 'xbw_xcolor_xbkg_unif_shdw_scatter_gauss_40_color_bias{}_model5_v{}'.format(color_pro*25.5, color_pro+1)
#        if color_pro == 0:
#            cmt = 'xbsw_xwing_scatter_gauss_30_bias0_model1_v1'
#        else:
#            cmt = 'xbsw_xwing_scatter_gauss_30_color_bias{}_model1_v{}'.format(color_pro*25.5, color_pro+1)
#        if color_pro == 0:
#            cmt = 'xbw_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_rnd_color_bias0_model4_v21'
#        else:
#            cmt = 'xbw_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_rnd_color_bias{}_model4_v{}'.format(color_pro*25.5, color_pro+21)
#        
#    for angle_pro in range(1, 11):
#        cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias{}_model4_v{}'.format(angle_pro*36, angle_pro+9)

#    for color_pro in range(11):
#        if color_pro == 0:
#            cmt = 'xbw_xcolor_xbkg_unif_shdw_scatter_gauss_38_color_bias0_model5_RC4_v1'
#        else:
#            cmt = 'xbw_xcolor_xbkg_unif_shdw_scatter_gauss_38_color_bias{}_model5_RC4_v{}'.format(color_pro*25.5, color_pro+1)
#        if color_pro == 0:
#            cmt = 'xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_1_color_bias0_model5_RC4_v12'
#        else:
#            cmt = 'xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_1_color_bias{}_model5_RC4_v{}'.format(color_pro*25.5, color_pro+12)
#        if color_pro == 0:
#            cmt = 'xbw_xcolor_xbkg_unif_shdw_scatter_gauss_50_color_bias0_model5_RC5_v1'
#        else:
#            cmt = 'xbw_xcolor_xbkg_unif_shdw_scatter_gauss_50_color_bias{}_model5_RC5_v{}'.format(color_pro*25.5, color_pro+1)
#
#        if color_pro == 0:
#            cmt = 'xbw_xbkg_unif_mig21_shdw_scatter_uniform_50_rndangle_rndsolar_color_bias0_RC1_v32'
#        else:
#            cmt = 'xbw_xbkg_unif_mig21_shdw_scatter_uniform_50_rndangle_rndsolar_color_bias{}_RC1_v{}'.format(color_pro*25.5, color_pro+32)
#        px_thres = 15
#    for color_pro in range(11):
#        if color_pro == 0:
#            cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_color_bias0_RC2_v12'
#        else:
#            cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_color_bias{}_RC2_v{}'.format(color_pro*25.5, color_pro+12)
#        px_thres = 23

#    for size_pro in range(6):
#        cmt = 'xbw_xbkg_unif_mig21_shdw_scatter_gauss_2_7_rndsolar_dynmu_size_bias{}_RC1_v{}'.format(size_pro*5, size_pro+43)
#        cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_fixedangle_dynmu_size_bias{}_RC1_v{}'.format(size_pro*5, size_pro+110)
#        px_thres = 15
#    for size_pro in range(6):
#        cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_dynmu_size_bias{}_RC1_v{}'.format(size_pro*5, size_pro+140)
#        px_thres = 15

#    for ix, size_pro in enumerate([-2, -1, 0, 1, 1.5, 2]):
#        cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_dynmu_size_bias{}_RC2_v{}'.format(10*size_pro, ix+23)
#        px_thres = 23

#    for ix, size_pro in enumerate(range(-1, 5)):
#        cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_size_bias{}_RC3_v{}'.format(7*size_pro, ix+23)
#        px_thres = 23

#    for ix, size_pro in enumerate([-3, -2, -1, 0, 1, 2]):
#        cmt = 'xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_size_bias{}_RC4_v{}'.format(size_pro*5, ix+23)
#        px_thres = 23

#    for ix, size_pro in enumerate(range(6)):
#        cmt = 'xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_size_bias{}_RC5_v{}'.format(size_pro*5, ix+12)
#        px_thres = 23

#    for ix, color_pro in enumerate([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]):
#        cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias{}_RC1_v{}'.format(color_pro, ix+50)
#        cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_fixedangle_dynmu_color_bias{}_RC1_v{}'.format(color_pro, ix+90)
#        px_thres = 15

#    for ix, color_pro in enumerate([0.5, -0.5, -1.5, -2.5, -3.5, -4.5]):
#        cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias{}_RC2_v{}'.format(color_pro, ix+30)
#        px_thres = 23

#    for ix, color_pro in enumerate([0, -0.5, -1.5, -2.5, -3.5, -4.5]):
#        cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias{}_RC3_v{}'.format(color_pro, ix+30)
#        px_thres = 23

#    for ix, color_pro in enumerate([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]):
#        cmt = 'xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias{}_RC4_v{}'.format(color_pro, ix+30)
#        px_thres = 23

#    for ix, color_pro in enumerate([-2.5, -1.5, 0, 0.5, 1.5, 2.5]):
#        cmt = 'xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias{}_RC5_v{}'.format(color_pro, ix+20)
#        px_thres = 23
   
#    bias_steps = [0, 0.2, 0.4, 0.6, 0.8]
#    for ix, color_pro in enumerate(bias_steps):
#        cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_dynsigma_color_bias{}_RC1_v{}'.format(color_pro, ix+60)
##        cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_light_dynsigma_color_bias{}_RC1_v{}'.format(color_pro, ix+80)
#        cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_fixedangle_dynsigma_color_bias{}_RC1_v{}'.format(color_pro, ix+100)
#        px_thres = 15

#    bias_steps = [0, 0.2, 0.4, 0.6, 0.8]
#    for ix, color_pro in enumerate(bias_steps):
#        cmt ='xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_dynsigma_color_bias{}_RC2_v{}'.format(color_pro, ix+40)
#        px_thres = 23
    
#    bias_steps = [0, 0.2, 0.4, 0.6, 0.8]
#    for ix, color_pro in enumerate(bias_steps):
#        cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynsigma_color_bias{}_RC3_v{}'.format(color_pro, ix+40)
#        px_thres = 23

#    bias_steps = [0, 0.2, 0.4, 0.6, 0.8]
#    for ix, color_pro in enumerate(bias_steps):
#        cmt = 'xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynsigma_color_bias{}_RC4_v{}'.format(color_pro, ix+40)
#        px_thres = 23
 
#    bias_steps = [0, 0.2, 0.4, 0.6, 0.8]
#    for ix, color_pro in enumerate(bias_steps):
#        cmt = 'xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynsigma_color_bias{}_RC5_v{}'.format(color_pro, ix+30)
#        px_thres = 23
    
#    for ix, size_pro in enumerate([0, 0.2, 0.4, 0.6]):
#        cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_dynsigma_size_bias{}_RC2_v{}'.format(size_pro, ix+50)
#        px_thres = 23
        
#        cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynsigma_size_bias{}_RC3_v{}'.format(size_pro, ix+50)
#        px_thres = 23
        
#        cmt = 'xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynsigma_size_bias{}_RC4_v{}'.format(size_pro, ix+50)
#        px_thres = 23

#    for ix, size_pro in enumerate([0, 0.05, 0.1, 0.15]):
#        cmt = 'xbw_xbkg_unif_mig21_shdw_scatter_gauss_2_7_rndsolar_dynsigma_size_bias{}_RC1_v{}'.format(size_pro, ix+70)
#        cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_fixedangle_dynsigma_size_bias{}_RC1_v{}'.format(size_pro, ix+120)
#        cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_dynsigma_size_bias{}_RC1_v{}'.format(size_pro, ix+140)
#        px_thres = 15
#        cmt = 'xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynsigma_size_bias{}_RC5_v{}'.format(size_pro, ix+140)
#        px_thres = 23

#    for i in range(1):
#        cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias-0.5_RC1_v52'
#        px_thres = 15

#        cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias0.5_RC2_v30'
#        px_thres = 23

#        cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias0.5_RC3_v30'
#        px_thres = 23

#        cmt = 'xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias-0.5_RC4_v31'
#        px_thres = 23

#        cmt = 'xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias-0.5_RC5_v22'
#        px_thres = 23

#        cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias-4.5_RC2_v35'
 #       px_thres = 23
    size_squsigma = [0, 0.03, 0.06, 0.09, 0.12]
    for ix,ssig in enumerate(size_squsigma):
        cmt = 'syn_xview_bkg_xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_bxmuller_size_bias{}_RC2_v{}'.format(ssig, ix+1)
        px_thres = 23


        whr_thres = 3
        dt = 'color'
        syn_args = get_args(cmt)
        group_object_annotation_and_draw_bbox(dt, px_thres, whr_thres)
        draw_bbx_on_rgb_images(dt, px_thres, whr_thres)

#    comments = ['xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_bias0_model4_v1',
#    'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_100_bias0_model4_v2',
#    'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias20_model4_v3',
#    'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias40_model4_v4',
#    'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias50_model4_v5',
#    'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias60_model4_v6',
#    'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias70_model4_v7',
#    'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias110_model4_v8',
#    'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_rnd_model4_v9']
#    for cmt in comments:
#        px_thres = 15
#        whr_thres = 3
#        dt = 'color'
#        syn_args = get_args(cmt)
#        group_object_annotation_and_draw_bbox(dt, px_thres, whr_thres)

#    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_bias0_model4_v1'
#    cmt = 'xbw_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_rnd_color_bias0_model4_v21'
#    cmt = 'xbw_xcolor_xbkg_unif_shdw_scatter_gauss_50_color_bias0_model5_RC5_v1'
#    px_thres = 30
#    whr_thres=3
#    display_types = ['color']# , 'mixed'] 'color',
#    syn_args = get_args(cmt)
#    for dt in display_types:
#        group_object_annotation_and_draw_bbox(dt, px_thres, whr_thres, upscale=True)

    '''
    draw bbox on rgb images for syn_xveiw_background data
    '''
    # # px_thres=6
    # # whr_thres=4
    # # px_thres=20
    # # whr_thres=4
    # px_thres=23
    # whr_thres=4
    # # display_types = ['texture', 'color', 'mixed']
    # # cmt = 'certain_models'
    # # cmt = 'scale_models'
    # cmt = 'small_models'
    # # cmt = 'small_fw_models'

    # cmt = '6groups_models'
    # cmt = 'rnd_bwratio_models'
    # cmt = 'rnd_bwratio_flat0.8_models'
    # cmt = 'rnd_bwratio_asx_models'
    # cmt = 'xratio_xcolor_models'
    # cmt = 'sbwratio_xratio_xcolor_models'
    # cmt = 'sbwratio_new_xratio_xcolor_models'
    # cmt = 'xbw_xcolor_model0'
    # # cmt = 'xbw_xrxc_spr_sml_models'

    # cmt = 'sbw_xcolor_model4'
    # cmt = 'sbw_xcolor_model4_v1'
    # cmt = 'sbw_xcolor_model4_v2'
    # cmt = 'xbw_xcolor_xbkg_gauss_model4_v3'
    # cmt = 'xbw_xcolor_xbkg_gauss_model4_v4'
    # cmt = 'xbw_xcolor_xbkg_unif_mig21_model4_v7'
    #    cmt = 'xbw_xcolor_xbkg_unif_mig21_rndp_model4_v7'
    #    cmt = 'xbw_rndcolor_xbkg_unif_mig21_model4_v8'
    #    cmt = 'xbw_rndcolor_xbkg_unif_mig21_rndp_model4_v9'
    #    cmt = 'xbw_xcolor_xbkg_unif_mig21_rndp_shdw_model4_v10'
    #    cmt = 'xbw_rndcolor_xbkg_unif_mig21_rndp_shdw_model4_v11'
    #    cmt = 'xbw_xcolor_xbkg_unif_mig21_rndp_shdw_dense_model4_v12'
    #    cmt = 'xbw_xcolor_xbkg_unif_mig21_rndp_shdw_dense_angle_model4_v13'
    #    cmt = 'xbw_xcolor_xbkg_unif_mig21_rndp_shdw_dense_angle_fwc_model4_v14'
    #    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_angle_fwc_scatter_gauss_dense_model4_v15'
    #    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_angle_fwc_scatter_gauss_dense_model4_v16'
    #    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_gauss_dense_model4_v17'
    #    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_gauss_dense_solar_model4_v18'
#        cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_uniform_50_model4_v19'
#    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_gauss_50_model4_v20'
#    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_gauss_120_model4_v21'
#    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_uniform_50_rdegree_model4_v22'
#    cmt = 'xbw_rndcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_model4_v23'
#    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_bias0_model4_v1'
#    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_100_bias0_model4_v2'
#    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias20_model4_v3'
#    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias40_model4_v4'
#    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias50_model4_v9'
#    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias60_model4_v8'
#    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias70_model4_v5'
#    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias110_model4_v6'
#    cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_rnd_model4_v7'
#    px_thres = 15
#    px_thres = 30
#    px_thres=50
    # cmt = 'sbw_xcolor_model0'
    # cmt = 'sbw_xcolor_model1'
    # cmt = 'xbw_xrxc_spr_sml_models_gauss'
    # cmt = 'xbw_xrxc_gauss_model1_v1'
    # cmt = 'xbw_xcolor_xbkg_gauss_model1_v2'
    # cmt = 'sbw_xcolor_xbkg_unif_model1_v3'
    # cmt = 'xbsw_xcolor_xbkg_gauss_model1_v4'
#    cmt = 'xbsw_xcolor_xbkg_unif_rndp_model1_v6'
#    cmt = 'xbsw_rndcolor_xbkg_unif_rndp_model1_v7'
#    cmt = 'xbsw_xwing_color_xbkg_unif_rndp_model1_v8'
#    cmt = 'xbsw_xwing_color_xbkg_unif_rndp_shdw_model1_v9'
#    px_thres=23 #20 #30
#    cmt = 'xbw_xcolor_xbkg_unif_shdw_scatter_uniform_model5_v1'
#    cmt = 'xbw_xcolor_xbkg_unif_shdw_rndp_model5_v2'
#    cmt = 'xbw_xcolor_xbkg_unif_shdw_scatter_gauss_40_bias0_model5_v1'
    
#    for color_pro in range(11):
##        if color_pro == 0:
##            cmt = 'xbw_xcolor_xbkg_unif_shdw_scatter_gauss_40_bias0_model5_v1'
##        else:
##            cmt = 'xbw_xcolor_xbkg_unif_shdw_scatter_gauss_40_color_bias{}_model5_v{}'.format(color_pro*25.5, color_pro+1)
##        if color_pro == 0:
##            cmt = 'xbsw_xwing_scatter_gauss_30_bias0_model1_v1'
##        else:
##            cmt = 'xbsw_xwing_scatter_gauss_30_color_bias{}_model1_v{}'.format(color_pro*25.5, color_pro+1)
##        if color_pro == 0:
##            cmt = 'xbw_xcolor_xbkg_unif_shdw_scatter_gauss_38_color_bias0_model5_RC4_v1'
##        else:
##            cmt = 'xbw_xcolor_xbkg_unif_shdw_scatter_gauss_38_color_bias{}_model5_RC4_v{}'.format(color_pro*25.5, color_pro+1)
#        if color_pro == 0:
#                cmt = 'xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_1_color_bias0_model5_RC4_v12'
#        else:
#            cmt = 'xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_1_color_bias{}_model5_RC4_v{}'.format(color_pro*25.5, color_pro+12)
#        px_thres = 23
##        if color_pro == 0:
##            cmt = 'xbw_xcolor_xbkg_unif_shdw_scatter_gauss_50_color_bias0_model5_RC5_v1'
##        else:
##            cmt = 'xbw_xcolor_xbkg_unif_shdw_scatter_gauss_50_color_bias{}_model5_RC5_v{}'.format(color_pro*25.5, color_pro+1)
##        px_thres = 15
#
#        whr_thres = 3
#        dt = 'color'
#        syn_args = get_args(cmt)
#        draw_bbx_on_rgb_images(dt, px_thres, whr_thres)
#
#        if color_pro == 0:
#            cmt = 'xbw_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_rnd_color_bias0_model4_v21'
#        else:
#            cmt = 'xbw_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_rnd_color_bias{}_model4_v{}'.format(color_pro*25.5, color_pro+21)
#        px_thres = 15
#
##    for angle_pro in range(11):
##        cmt = 'xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias{}_model4_v{}'.format(angle_pro*36, angle_pro+9)
##        px_thres = 15
#
#        whr_thres = 3
#        dt = 'color'
#        syn_args = get_args(cmt)
#        draw_bbx_on_rgb_images(dt, px_thres, whr_thres)


#    px_thres = 23
#    whr_thres=3
#    display_types = ['color']# , 'mixed']
#    syn_args = get_args(cmt)
#    for dt in display_types:
#        draw_bbx_on_rgb_images(dt, px_thres, whr_thres)

#    bias_steps = [0, 0.2, 0.4, 0.6, 0.8]
#    for ix, color_pro in enumerate(bias_steps):
#        cmt = 'xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynsigma_color_bias{}_RC4_v{}'.format(color_pro, ix+60)
#        px_thres = 23
#        cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_light_dynsigma_color_bias{}_RC1_v{}'.format(color_pro, ix+80)
#        px_thres = 15
#
##    for size_pro in range(6):
##        cmt = 'xbw_xbkg_unif_mig21_shdw_scatter_gauss_2_7_rndsolar_dynmu_size_bias{}_RC1_v{}'.format(size_pro*5, size_pro+43)
##        px_thres = 15
#
#    for ix, size_pro in enumerate([0, 0.05, 0.1, 0.15]):
##        cmt = 'xbw_xbkg_unif_mig21_shdw_scatter_gauss_2_7_rndsolar_dynsigma_size_bias{}_RC1_v{}'.format(size_pro, ix+70)
#        cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_fixedangle_dynsigma_size_bias{}_RC1_v{}'.format(size_pro, ix+120)
#        px_thres = 15
#        
##    for ix, size_pro in enumerate([-3, -2, -1, 0, 1, 2]):
##        cmt = 'xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_size_bias{}_RC4_v{}'.format(size_pro*5, ix+23)
##        px_thres = 23
##    for size_pro in range(6):
##        cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_fixedangle_dynmu_size_bias{}_RC1_v{}'.format(size_pro*5, size_pro+110)
#
#        whr_thres = 3
#        dt = 'color'
#        syn_args = get_args(cmt)
#        draw_bbx_on_rgb_images(dt, px_thres, whr_thres)




