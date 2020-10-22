import glob
import numpy as np
import argparse
import os
import shutil
from utils.object_score_util import get_bbox_coords_from_annos_with_object_score as gbc
from PIL import ImageColor

def get_lbl_for_CC_plot_bbx_for_CC(label_index = True, label_id = False):
    '''
    munually select images of CC
    backup corresponding original labels for CC
    plot bbx on images
    manually label the bbox with CCid
    :return:
    '''
    cc1_img_dir = os.path.join(args.images_save_dir, '{}_{}cls_cc1'.format(args.input_size, args.class_num))
    cc2_img_dir = os.path.join(args.images_save_dir, '{}_{}cls_cc2'.format(args.input_size, args.class_num))
    lbl_dir = os.path.join(args.annos_save_dir, '{}_cls_xcycwh_px{}whr{}_with_id'.format(args.class_num, px_thres, whr_thres))

    cc1_lbl_dir = os.path.join(args.annos_save_dir, '{}_cls_xcycwh_px{}whr{}_cc1_with_id'.format(args.class_num, px_thres, whr_thres))
    len_lbl_cc1 = len(glob.glob(os.path.join(cc1_lbl_dir, '*.txt')))
    cc2_lbl_dir = os.path.join(args.annos_save_dir, '{}_cls_xcycwh_px{}whr{}_cc2_with_id'.format(args.class_num, px_thres, whr_thres))
    len_lbl_cc2 = len(glob.glob(os.path.join(cc2_lbl_dir, '*.txt')))

    if not os.path.exists(cc1_lbl_dir):
        os.mkdir(cc1_lbl_dir)

    if not os.path.exists(cc2_lbl_dir):
        os.mkdir(cc2_lbl_dir)

    cc1_fig_dir = os.path.join(args.cat_sample_dir, 'image_with_bbox_indices', 'CC1')
    if not os.path.exists(cc1_fig_dir):
        os.mkdir(cc1_fig_dir)

    cc2_fig_dir = os.path.join(args.cat_sample_dir, 'image_with_bbox_indices', 'CC2')
    if not os.path.exists(cc2_fig_dir):
        os.mkdir(cc2_fig_dir)

    cc1_img_files = glob.glob(os.path.join(cc1_img_dir, '*.jpg'))
    cc2_img_files = glob.glob(os.path.join(cc2_img_dir, '*.jpg'))


    for f in cc1_img_files:
        lbl_name = os.path.basename(f).replace('.jpg', '.txt')
        if not len(len_lbl_cc1):
            shutil.copy(os.path.join(lbl_dir, lbl_name),
                        os.path.join(cc1_lbl_dir, lbl_name))
        gbc.plot_img_with_bbx(f, os.path.join(cc1_lbl_dir, lbl_name), cc1_fig_dir, label_index, label_id)

    for f in cc2_img_files:
        lbl_name = os.path.basename(f).replace('.jpg', '.txt')
        if not len(len_lbl_cc2):
            shutil.copy(os.path.join(lbl_dir, lbl_name),
                    os.path.join(cc2_lbl_dir, lbl_name))
        gbc.plot_img_with_bbx(f, os.path.join(cc2_lbl_dir, lbl_name), cc2_fig_dir,  label_index, label_id)


def compute_rgb_by_hex(hex):
    hex_list = [s for s in hex.split(';')]
    r_list = []
    g_list = []
    b_list = []
    for hex in hex_list:
        (r, g, b) = ImageColor.getcolor(hex, "RGB")
        r_list.append(r)
        g_list.append(g)
        b_list.append(b)
    r_mean = np.round(np.mean(r_list)).astype(np.int)
    g_mean = np.round(np.mean(g_list)).astype(np.int)
    b_mean = np.round(np.mean(b_list)).astype(np.int)
    rgb_mean = np.array([r_mean, g_mean, b_mean])
    print('rgb mean', rgb_mean)


def get_args(px_thres=None, whr_thres=None, seed=17):
    parser = argparse.ArgumentParser()

    parser.add_argument("--xview_yolo_dir", type=str, help="dir to xViewYOLO",
                        default='/media/lab/Yang/data/xView_YOLO/')

    parser.add_argument("--images_save_dir", type=str, help="to save chip trn val images files",
                        default='/media/lab/Yang/data/xView_YOLO/images/')

    parser.add_argument("--annos_save_dir", type=str, help="to save txt annotation files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/')

    parser.add_argument("--txt_save_dir", type=str, help="to save  related label files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')

    parser.add_argument("--data_save_dir", type=str, help="to save data files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_cls/')
    parser.add_argument("--cat_sample_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/')

    parser.add_argument("--font3", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 10}")
    parser.add_argument("-ft2", "--font2", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 13}")
    parser.add_argument("--class_num", type=int, default=1, help="Number of Total Categories")  # 60  6 1
    parser.add_argument("--input_size", type=int, default=608, help="Number of Total Categories")  # 300 416

    args = parser.parse_args()
    #fixme ----------==--------change
    args.data_save_dir = args.data_save_dir.format(args.class_num)
    args.annos_save_dir = args.annos_save_dir.format(args.input_size)
    args.txt_save_dir = args.txt_save_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)
    args.cat_sample_dir = args.cat_sample_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)

    if not os.path.exists(args.txt_save_dir):
        os.makedirs(args.txt_save_dir)

    if not os.path.exists(args.annos_save_dir):
        os.makedirs(args.annos_save_dir)

    if not os.path.exists(args.cat_sample_dir):
        os.makedirs(args.cat_sample_dir)

    return args


if __name__ == '__main__':
    px_thres = 23
    whr_thres = 3
    args = get_args(px_thres, whr_thres)

    '''
    munually select images of CC
    backup corresponding original labels for CC
    plot bbx on images
    manually label the bbox with CCid
    '''
    # get_lbl_for_CC_plot_bbx_for_CC(label_index=True, label_id=False)

    # cc1_lbl_dir = os.path.join(args.annos_save_dir, '{}_cls_xcycwh_px{}whr{}_cc1_with_id'.format(args.class_num, px_thres, whr_thres))
    # lbl_cc1_names = [os.path.basename(f) for f in glob.glob(os.path.join(cc1_lbl_dir, '*.txt'))]
    # cc2_lbl_dir = os.path.join(args.annos_save_dir, '{}_cls_xcycwh_px{}whr{}_cc2_with_id'.format(args.class_num, px_thres, whr_thres))
    # lbl_cc2_names = [os.path.basename(f) for f in glob.glob(os.path.join(cc2_lbl_dir, '*.txt'))]
    #
    # for n in lbl_cc1_names:
    #     if n in lbl_cc2_names:
    #         shutil.copy(os.path.join(cc1_lbl_dir, n), os.path.join(cc2_lbl_dir, n))

    '''
    CC1 color
    # rgb mean [214 206 199]
    '''
    cc1_hexes = '#cc8d61;#f8f2f5;#b2b0ad;#faeeeb;#f6f3e8;#fffff9;#fbfafb;#fcfcf9;#feecda;#b7b1b0;#c6bdab;#d9d1bc;#d0c7b7;#dcd2c1;#d5cabc;#fefef9;#f9f6f0;#fdf4e8;#62615c;#585850;' \
                     '#ccc6d0;#cec9cf;#d3d0cd;#d6cdcb;#f9f4f0;#e3e0db;#dfd9dc;#c0b9b3;#e5eaf0;#e2d8d4;#d1d6e7;#c7cadc;#d1ccd9;#c4d4dc;#cbc6cb;#cfd0d9;#eff0ef;#9ca6ad;#b9987e;#feece5;' \
                     '#f5e0d3;#f8e0bf;#ffeecb;#f9eadd;#d79d6f;#feeac8;#b9b0a6;#cdcac4;#e2dedb;#a1a8a7;#d3bfba;#9e9c9a;#9ea19f'
    compute_rgb_by_hex(cc1_hexes)

    '''
    CC2 color
    # rgb mean [235 229 224]
    # rgb mean [202 202 199]
    '''
    cc2_hexes = '#d9d0ce;#afafba;#f0eae3;#fef6e4;#ddd3d0;#f8ece5;#e6e3d5;#ddcdc6;#e7d6bf;#e8d8c6;#cabab7;#eaf2ee;#fffbf5;#fffaf0;' \
                '#cfcfcb;#fff2ec;#cec4b8;#f9f5ed;#dcdbd7;#d2c9c2;#fbfaf6;#fbfaf6;#fefdf8;#e9e7df;#bcbdbf;#eaeaec;#fff9fa;#e0e2e1;' \
                '#fffbef;#fefaf9;#e8dfd6;#fffefc;#fef9f6;#feffff;#fffbff;#fdf3e9;#c5bfc1;#e9e8f0;#fff0e2'
    cc2_wing_hexes = '#d5d2d1;#d7d8cf;#b2b4b1;#d6d7ce;#c9c8c6;#cbd0d8;#b7b9b6;#dcd3d0;#cbc7c0;#b4b4b2;#d7d9d7;#d3dbd8;' \
                     '#d3dbd8;#767c86;#b6b6a2;#b1b2a1' ## #f1dedd;#fdfcfa;
    # compute_rgb_by_hex(cc2_hexes)
    compute_rgb_by_hex(cc2_wing_hexes)


