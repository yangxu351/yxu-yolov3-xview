import glob
import numpy as np
import argparse
import os
import pandas as pd
from ast import literal_eval
from matplotlib import pyplot as plt
import cv2
import json
import shutil
from utils.data_process_distribution_vis_util import process_wv_coco_for_yolo_patches_no_trnval as pwv
from utils.xview_synthetic_util import preprocess_synthetic_data_distribution as pps
from utils.object_score_util import get_bbox_coords_from_annos_with_object_score as gbc
from utils.utils_xview import coord_iou

IMG_SUFFIX = '.png'
TXT_SUFFIX = '.txt'


def check_difference_of_first_second_dataset():
    new_val = pd.read_csv('/data_xview/1_cls/second_dataset_backup/xviewval_img.txt', header=None)
    new_val = list(new_val.loc[:, 0])
    new_vname = [os.path.basename(f) for f in new_val]
    new_val_lbl = pd.read_csv('/data_xview/1_cls/second_dataset_backup/xviewval_lbl.txt', header=None)
    new_val_lbl = list(new_val_lbl.loc[:, 0])

    new_trn = pd.read_csv('/data_xview/1_cls/second_dataset_backup/xviewtrain_img.txt', header=None)
    new_trn = list(new_trn.loc[:, 0])
    new_tname = [os.path.basename(f) for f in new_trn]
    new_trn_lbl = pd.read_csv('/data_xview/1_cls/second_dataset_backup/xviewtrain_lbl.txt', header=None)
    new_trn_lbl = list(new_trn_lbl.loc[:, 0])

    old_val = pd.read_csv('/media/lab/Yang/code/yolov3/data_xview/1_cls/first_data_set_backup/xview_val_img.txt', header=None)
    old_val = list(old_val.loc[:, 0])
    old_vname = [os.path.basename(f) for f in old_val]
    old_val_lbl = pd.read_csv('/media/lab/Yang/code/yolov3/data_xview/1_cls/first_data_set_backup/xview_val_lbl.txt', header=None)
    old_val_lbl = list(old_val_lbl.loc[:, 0])

    old_trn = pd.read_csv('/media/lab/Yang/code/yolov3/data_xview/1_cls/first_data_set_backup/xview_train_img.txt', header=None)
    old_trn = list(old_trn.loc[:, 0])
    old_tname = [os.path.basename(f) for f in old_trn]
    old_trn_lbl = pd.read_csv('/media/lab/Yang/code/yolov3/data_xview/1_cls/first_data_set_backup/xview_train_lbl.txt', header=None)
    old_trn_lbl = list(old_trn_lbl.loc[:, 0])

    only_old_v = [v for v in old_vname if v not in new_vname]
    only_old_t = [t for t in old_tname if t not in new_tname]
    only_new_v = [v for v in new_vname if v not in old_vname]
    only_new_t = [t for t in new_tname if t not in old_tname]
    print('only_old_v', len(only_old_v))
    print('only_old_t', len(only_old_t))
    print('only_new_v', len(only_new_v))
    print('only_new_t', len(only_new_t))

    data_dir = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_drop/'
    only_old_t_dir = os.path.join(data_dir, 'only_in_old_trn')
    if not os.path.exists(only_old_t_dir):
        os.mkdir(only_old_t_dir)
    only_old_v_dir = os.path.join(data_dir, 'only_in_old_val')
    if not os.path.exists(only_old_v_dir):
        os.mkdir(only_old_v_dir)
    only_new_t_dir = os.path.join(data_dir, 'only_in_new_trn')
    if not os.path.exists(only_new_t_dir):
        os.mkdir(only_new_t_dir)
    only_new_v_dir = os.path.join(data_dir, 'only_in_new_val')
    if not os.path.exists(only_new_v_dir):
        os.mkdir(only_new_v_dir)

    for v in only_old_t:
        # shutil.copy(old_trn[old_tname.index(v)], os.path.join(only_old_t_dir, v))
        gbc.plot_img_with_bbx(old_trn[old_tname.index(v)], old_trn_lbl[old_tname.index(v)], only_old_t_dir)
    for v in only_old_v:
        # shutil.copy(old_val[old_vname.index(v)], os.path.join(only_old_v_dir, v))
        gbc.plot_img_with_bbx(old_val[old_vname.index(v)], old_val_lbl[old_vname.index(v)], only_old_v_dir)
    for v in only_new_t:
        # shutil.copy(new_trn[new_tname.index(v)], os.path.join(only_new_t_dir, v))
        gbc.plot_img_with_bbx(new_trn[new_tname.index(v)], new_trn_lbl[new_tname.index(v)], only_new_t_dir)
    for v in only_new_v:
        # shutil.copy(new_val[new_vname.index(v)], os.path.join(only_new_v_dir, v))
        gbc.plot_img_with_bbx(new_val[new_vname.index(v)], new_val_lbl[new_vname.index(v)], only_new_v_dir)


def combine_xview_syn_by_ratio(syn_ratio):
    syn_args = pps.get_syn_args()
    args = pwv.get_args()
    # print(os.path.join(syn_args.data_save_dir, '{}_{}_img.txt'.format(syn_args.syn_display_type, syn_args.class_num)))
    syn_img_txt = pd.read_csv(open(os.path.join(syn_args.syn_data_list_dir, '{}_{}_img.txt'.format(syn_args.syn_display_type, syn_args.class_num))), header=None).to_numpy()
    syn_lbl_txt = pd.read_csv(open(os.path.join(syn_args.syn_data_list_dir, '{}_{}_lbl.txt'.format(syn_args.syn_display_type, syn_args.class_num))), header=None).to_numpy()
    syn_total_num = syn_img_txt.shape[0]
    np.random.seed(syn_args.seed)
    perm_indices = np.random.permutation(syn_total_num)
    syn_ratio_num = np.int(syn_total_num*syn_ratio)

    xview_img_txt = pd.read_csv(open(os.path.join(args.data_save_dir, 'xviewtrain_img.txt')), header=None).to_numpy()
    xview_lbl_txt = pd.read_csv(open(os.path.join(args.data_save_dir, 'xviewtrain_lbl.txt')), header=None).to_numpy()
    xview_trn_num = xview_img_txt.shape[0]

    xview_syn_trn_txt_list_dir = os.path.join(args.data_save_dir, 'xview_{}_{}'.format(syn_args.syn_display_type, syn_ratio))
    if not os.path.exists(xview_syn_trn_txt_list_dir):
        os.mkdir(xview_syn_trn_txt_list_dir)

    f_img = open(os.path.join(xview_syn_trn_txt_list_dir, 'xview_{}_{}_train_img.txt'.format(syn_args.syn_display_type, syn_ratio)), 'w')
    f_lbl = open(os.path.join(xview_syn_trn_txt_list_dir, 'xview_{}_{}_train_lbl.txt'.format(syn_args.syn_display_type, syn_ratio)), 'w')

    for ix in range(xview_trn_num):
        f_img.write("%s\n" % xview_img_txt[ix, 0])
        f_lbl.write("%s\n" % xview_lbl_txt[ix, 0])

    for i in range(syn_ratio_num):
        f_img.write("%s\n" % syn_img_txt[perm_indices[i], 0])
        f_lbl.write("%s\n" % syn_lbl_txt[perm_indices[i], 0])
    f_img.close()
    f_lbl.close()

    xview_syn_data_list_dir = os.path.join(args.data_list_save_dir, 'xview_{}_{}'.format(syn_args.syn_display_type, syn_ratio))
    if not os.path.exists(xview_syn_data_list_dir):
        os.mkdir(xview_syn_data_list_dir)
    shutil.copy(os.path.join(xview_syn_trn_txt_list_dir, 'xview_{}_{}_train_img.txt'.format(syn_args.syn_display_type, syn_ratio)),
                os.path.join(xview_syn_data_list_dir, 'xview_{}_{}_train_img.txt'.format(syn_args.syn_display_type, syn_ratio)))
    shutil.copy(os.path.join(xview_syn_trn_txt_list_dir, 'xview_{}_{}_train_lbl.txt'.format(syn_args.syn_display_type, syn_ratio)),
                os.path.join(xview_syn_data_list_dir, 'xview_{}_{}_train_lbl.txt'.format(syn_args.syn_display_type, syn_ratio)))


def mv_list():
    '''
    move the *.txt to des dir if necessary
    :return:
    '''
    syn_args = pps.get_syn_args()
    args = pwv.get_args()
    srs = [0.25, 0.5, 0.75]
    for syn_ratio in srs:
        xview_syn_trn_txt_list_dir = os.path.join(args.data_save_dir, 'xview_{}_{}'.format(syn_args.syn_display_type, syn_ratio))
        if not os.path.exists(xview_syn_trn_txt_list_dir):
            os.mkdir(xview_syn_trn_txt_list_dir)
        xview_syn_data_list_dir = os.path.join(args.data_list_save_dir, 'xview_{}_{}'.format(syn_args.syn_display_type, syn_ratio))
        if not os.path.exists(xview_syn_data_list_dir):
            os.mkdir(xview_syn_data_list_dir)
        f_img = os.path.join(args.data_save_dir, 'xview_{}_{}_train_img.txt'.format(syn_args.syn_display_type, syn_ratio))
        f_lbl = os.path.join(args.data_save_dir, 'xview_{}_{}_train_lbl.txt'.format(syn_args.syn_display_type, syn_ratio))
        f_data = os.path.join(args.data_save_dir, 'xview_{}_{}.data'.format(syn_args.syn_display_type, syn_ratio))
        t_img = os.path.join(xview_syn_trn_txt_list_dir, 'xview_{}_{}_train_img.txt'.format(syn_args.syn_display_type, syn_ratio))
        t_lbl = os.path.join(xview_syn_trn_txt_list_dir, 'xview_{}_{}_train_lbl.txt'.format(syn_args.syn_display_type, syn_ratio))
        t_data = os.path.join(xview_syn_trn_txt_list_dir, 'xview_{}_{}.data'.format(syn_args.syn_display_type, syn_ratio))
        if os.path.exists(f_img):
            shutil.move(f_img, t_img)
        if os.path.exists(f_lbl):
            shutil.move(f_lbl, t_lbl)
        if os.path.exists(f_data):
            shutil.move(f_data, t_data)

        xview_syn_data_list_dir = os.path.join(args.data_list_save_dir, 'xview_{}_{}'.format(syn_args.syn_display_type, syn_ratio))
        if not os.path.exists(xview_syn_data_list_dir):
            os.mkdir(xview_syn_data_list_dir)

        src_img_txt = os.path.join(args.data_list_save_dir, 'xview_{}_{}_train_img.txt'.format(syn_args.syn_display_type, syn_ratio))
        src_lbl_txt = os.path.join(args.data_list_save_dir, 'xview_{}_{}_train_lbl.txt'.format(syn_args.syn_display_type, syn_ratio))
        dst_img_txt = os.path.join(args.data_list_save_dir, 'xview_{}_{}'.format(syn_args.syn_display_type, syn_ratio), 'xview_{}_{}_train_img.txt'.format(syn_args.syn_display_type, syn_ratio))
        dst_lbl_txt = os.path.join(args.data_list_save_dir, 'xview_{}_{}'.format(syn_args.syn_display_type, syn_ratio), 'xview_{}_{}_train_lbl.txt'.format(syn_args.syn_display_type, syn_ratio))
        if os.path.exists(src_img_txt):
            shutil.move(src_img_txt, dst_img_txt)
        if os.path.exists(src_lbl_txt):
            shutil.move(src_lbl_txt, dst_lbl_txt)


def collect_syn_data(data_name):
    syn_args = pps.get_syn_args()
    images_save_dir = syn_args.syn_images_save_dir
    txt_save_dir = syn_args.syn_txt_save_dir
    data_save_dir = syn_args.syn_data_list_dir
    lbl_path = syn_args.syn_annos_save_dir

    all_files = glob.glob(images_save_dir + '*'+IMG_SUFFIX)
    num_files = len(all_files)

    img_txt = open(os.path.join(txt_save_dir, '{}_img.txt'.format(data_name)), 'w')
    lbl_txt = open(os.path.join(txt_save_dir, '{}_lbl.txt'.format(data_name)), 'w')

    for i in range(num_files):
        img_txt.write("%s\n" % all_files[i])
        img_name = all_files[i].split('/')[-1]
        lbl_name = img_name.replace(IMG_SUFFIX, TXT_SUFFIX)
        lbl_txt.write("%s\n" % (lbl_path + lbl_name))

    img_txt.close()
    lbl_txt.close()

    shutil.copyfile(os.path.join(txt_save_dir, '{}_img.txt'.format(data_name)),
                    os.path.join(data_save_dir, '{}_img.txt'.format(data_name)))
    shutil.copyfile(os.path.join(txt_save_dir, '{}_lbl.txt'.format(data_name)),
                    os.path.join(data_save_dir, '{}_lbl.txt'.format(data_name)))


def create_xview_syn_data(dt, sr):
    args = pwv.get_args()
    syn_args = pps.get_syn_args()
    if sr:
        data_txt = open(os.path.join(args.data_save_dir, 'xview_{}_{}'.format(dt, sr), 'xview_{}_{}.data'.format(dt, sr)), 'w')
        data_txt.write('train=./data_xview/{}_cls/xview_{}_{}/xview_{}_{}_train_img.txt\n'.format(syn_args.class_num, dt, sr, dt, sr))
        data_txt.write('train_label=./data_xview/{}_cls/xview_{}_{}/xview_{}_{}_train_lbl.txt\n'.format(syn_args.class_num, dt, sr, dt, sr))
    else: # sr==0
        dt = 'syn'
        data_txt = open(os.path.join(args.data_save_dir, 'xview_{}_{}.data'.format(dt, sr)), 'w')
        data_txt.write('train=./data_xview/{}_cls/xviewtrain_img.txt\n'.format(syn_args.class_num, dt, sr))
        data_txt.write('train_label=./data_xview/{}_cls/xviewtrain_lbl.txt\n'.format(syn_args.class_num, dt, sr))

    df = pd.read_csv(os.path.join(args.data_save_dir, 'xviewtrain_img.txt'), header=None)
    data_txt.write('syn_0_xview_number=%s\n' % str(df.shape[0]))
    data_txt.write('classes=%s\n' % str(syn_args.class_num))
    data_txt.write('valid=./data_xview/{}_cls/xviewval_img.txt\n'.format(syn_args.class_num))
    data_txt.write('valid_label=./data_xview/{}_cls/xviewval_lbl.txt\n'.format(syn_args.class_num))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(syn_args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval=xview')
    data_txt.close()


if __name__ == "__main__":

    args = pps.get_syn_args()

    '''
    create chips and label txt and get all images json, convert from *.geojson to *.json
    
    get all images json, convert from *.geojson to *.json
    # convert_geojson_to_json()
    create label txt files
    # create_label_txt()
    
    '''
    # pwv.create_chips_and_txt_geojson_2_json(syn = True)

    '''
    cnt and get overlap bbox & remove duplicate bbox
    '''

    # cat_ids = [0]
    # pwv.cnt_ground_truth_overlap_from_pathces(cat_ids, syn = True)

    '''
    remove duplicate ground truth bbox by cat_id
    then manually replace the old duplicate annotation file with the new annotation file ************
    '''
    # # #
    # cat_id = 0
    # pwv.remove_duplicate_gt_bbx(cat_id, syn = True)

    '''
    backup xview-plane dataset ***********
    clean xview-plane dataset with certain constraints
    '''
    # whr_thres = 4  # 3
    # px_thres = 6  # 4
    # pwv.clean_backup_xview_plane_with_constraints(px_thres, whr_thres)

    ''' if necessary
    check xview aireplane dropped images
    '''
    # pwv.check_xview_plane_drops()

    ''' if necessary
    recover xview val list with constriants of px_theres=4, whr_thres=3
    regenerate_xview_val_list
    '''
    # pwv.recover_xview_val_list()

    ''' if necessary
    check difference of first and second dataset
    '''
    # check_difference_of_first_second_dataset()

    '''
    create xview.names 
    '''
    # file_name = 'xview'
    # pwv.create_xview_names(file_name)

    '''
    xview
    split train:val randomly split chips
    default split 
    '''
    # data_name = 'xview'
    # pwv.split_trn_val_with_chips(data_name)

    '''
    collect all syn images and txt into one file
    '''
    # data_name = '{}_{}'.format(args.syn_display_type, args.class_num)
    # collect_syn_data(data_name)

    '''
    combine xview & synthetic dataset [0.25, 0.5, 0.75]
    change syn_display_type manually ***
    '''
    # syn_ratio = [0.25, 0.5, 0.75]
    # for sr in syn_ratio:
    #     combine_xview_syn_by_ratio(sr)

    '''
    move the *.txt to des dir if necessary
    '''
    # mv_list()

    ''''
    create xview_syn_texture_0.25.data
    '''
    # syn_ratio = [0.25, 0.5, 0.75]
    # display_type = ['syn_texture', 'syn_color', 'syn_mixed']
    # for dt in display_type:
    #     for sr in syn_ratio:
    #         create_xview_syn_data(dt, sr)

    # create_xview_syn_data('syn', 0)


