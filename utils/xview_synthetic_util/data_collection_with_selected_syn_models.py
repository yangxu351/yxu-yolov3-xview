import glob
import numpy as np
import os
import pandas as pd
import shutil
from utils.data_process_distribution_vis_util import process_wv_coco_for_yolo_patches_no_trnval as pwv
from utils.xview_synthetic_util import preprocess_synthetic_data_distribution as pps


def generate_new_xview_lbl_with_model_id(type='val', nonmatch=3):
    '''
    generate new xview train val annotations (with model id)
    generate new xview train val list
    :return:
    '''
    args = pwv.get_args()
    ori_val_lbl_txt = os.path.join(args.data_save_dir, 'xview{}_lbl.txt'.format(type))
    df_ori_val = pd.read_csv(ori_val_lbl_txt, header=None)
    ori_val_names = [os.path.basename(f) for f in df_ori_val.iloc[:, 0]]

    xview_val_lbl_with_model_name = 'xview{}_lbl_with_model.txt'.format(type)
    xview_val_lbl_with_model_txt = open(os.path.join(args.data_save_dir, xview_val_lbl_with_model_name), 'w')

    des_val_lbl_path = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_only_model/'
    if not os.path.exists(des_val_lbl_path):
        os.mkdir(des_val_lbl_path)

    src_lbl_path = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_model/'

    for i in range(len(ori_val_names)):
        f = os.path.join(src_lbl_path, ori_val_names[i])
        name = os.path.basename(f)
        if not pps.is_non_zero_file(f):
            shutil.copy(f, os.path.join(des_val_lbl_path, name))
            xview_val_lbl_with_model_txt.write('%s\n' % os.path.join(des_val_lbl_path, name))
            continue

        colomns = np.arange(0, 6)
        src_lbl = pd.read_csv(f, header=None, sep=' ', names=colomns)
        des_lbl_txt = open(os.path.join(des_val_lbl_path, name), 'w')

        for i in range(src_lbl.shape[0]):
            des_lbl_txt.write("%d %.8f %.8f %.8f %.8f %d\n" % (src_lbl.iloc[i, 0], src_lbl.iloc[i, 1], src_lbl.iloc[i, 2], src_lbl.iloc[i, 3], src_lbl.iloc[i, 4], src_lbl.iloc[i, 5] if not np.isnan(src_lbl.iloc[i, 5]) else nonmatch))
        des_lbl_txt.close()
        xview_val_lbl_with_model_txt.write('%s\n' % os.path.join(des_val_lbl_path, name))
    xview_val_lbl_with_model_txt.close()
    shutil.copy(os.path.join(args.data_save_dir, xview_val_lbl_with_model_name), os.path.join(args.data_list_save_dir, xview_val_lbl_with_model_name))


def generate_new_syn_lbl_with_model_id(comments='', nonmatch=3):
    '''
    generate new syn annotations (with model id)
    generate new syn  list
    :return:
    '''
    syn_args = pps.get_syn_args()
    ori_val_lbl_txt = os.path.join(syn_args.syn_data_list_dir, '{}_{}_lbl.txt'.format(syn_args.syn_display_type, syn_args.class_num))
    df_ori_val = pd.read_csv(ori_val_lbl_txt, header=None)
    ori_val_names = [os.path.basename(f) for f in df_ori_val.loc[:, 0]]

    xview_val_lbl_with_model_name = '{}_{}_lbl{}.txt'.format(syn_args.syn_display_type, syn_args.class_num, comments)
    xview_val_lbl_with_model_txt = open(os.path.join(syn_args.syn_data_list_dir, xview_val_lbl_with_model_name), 'w')

    des_val_lbl_path = '/media/lab/Yang/data/xView_YOLO/labels/608/{}_{}_cls_xcycwh_only_model/'.format(syn_args.syn_display_type, syn_args.class_num)
    if not os.path.exists(des_val_lbl_path):
        os.mkdir(des_val_lbl_path)

    src_lbl_path = '/media/lab/Yang/data/xView_YOLO/labels/608/{}_{}_cls_xcycwh_model/'.format(syn_args.syn_display_type, syn_args.class_num)

    for name in ori_val_names:
        src_lbl_file = os.path.join(src_lbl_path, name)
        if not pps.is_non_zero_file(src_lbl_file):
            shutil.copy(src_lbl_file, os.path.join(des_val_lbl_path, name))
            xview_val_lbl_with_model_txt.write('%s\n' % os.path.join(des_val_lbl_path, name))
            continue

        colomns = np.arange(0, 6)
        src_lbl = pd.read_csv(src_lbl_file, header=None, sep=' ', names=colomns)
        des_lbl_txt = open(os.path.join(des_val_lbl_path, name), 'w')

        for i in range(src_lbl.shape[0]):
            if np.isnan(src_lbl.iloc[i, -1]):
                des_lbl_txt.write("%d %.8f %.8f %.8f %.8f %d\n" % (src_lbl.iloc[i, 0], src_lbl.iloc[i, 1], src_lbl.iloc[i, 2], src_lbl.iloc[i, 3], src_lbl.iloc[i, 4], nonmatch))
            else:
                des_lbl_txt.write("%d %.8f %.8f %.8f %.8f %d\n" % (src_lbl.iloc[i, 0], src_lbl.iloc[i, 1], src_lbl.iloc[i, 2], src_lbl.iloc[i, 3], src_lbl.iloc[i, 4], src_lbl.iloc[i, 5]))
        des_lbl_txt.close()
        xview_val_lbl_with_model_txt.write('%s\n' % os.path.join(des_val_lbl_path, name))
    xview_val_lbl_with_model_txt.close()
    # shutil.copy(os.path.join(args.data_save_dir, xview_val_lbl_with_model_name), os.path.join(args.data_list_save_dir, xview_val_lbl_with_model_name))


if __name__ == '__main__':
    '''
    generate new xviewval_lbl_with_model.txt
    '''
    # type = 'train'
    # # type = 'val'
    # nonmatch = 3
    # generate_new_xview_lbl_with_model_id(type, nonmatch)

    '''
    generate new syn_*_lbl_with_model.txt
    '''
    # comments = '_with_model'
    # nonmatch = 3
    # generate_new_syn_lbl_with_model_id(comments, nonmatch)

