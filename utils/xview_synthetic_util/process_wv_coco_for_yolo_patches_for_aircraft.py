import glob
import numpy as np
import argparse
import os
import pandas as pd
from ast import literal_eval
from matplotlib import pyplot as plt
import json
import shutil
from utils.data_process_distribution_vis_util import process_wv_coco_for_yolo_patches_no_trnval as pwv
from utils.xview_synthetic_util import preprocess_synthetic_data_distribution as pps

IMG_SUFFIX = '.jpg'
TXT_SUFFIX = '.txt'


def combine_xview_syn_by_ratio(syn_ratio, separate=False):
    syn_args = pps.get_syn_args()
    args = pwv.get_args()
    # print(os.path.join(syn_args.data_save_dir, '{}_{}_img.txt'.format(syn_args.syn_display_type, syn_args.class_num)))
    syn_img_txt = pd.read_csv(open(os.path.join(syn_args.data_save_dir, '{}_{}_img.txt'.format(syn_args.syn_display_type, syn_args.class_num))), header=None).to_numpy()
    syn_lbl_txt = pd.read_csv(open(os.path.join(syn_args.data_save_dir, '{}_{}_lbl.txt'.format(syn_args.syn_display_type, syn_args.class_num))), header=None).to_numpy()
    syn_total_num = syn_img_txt.shape[0]
    np.random.seed(syn_args.seed)
    perm_indices = np.random.permutation(syn_total_num)
    syn_ratio_num = np.int(syn_total_num*syn_ratio)

    xview_img_txt = pd.read_csv(open(os.path.join(args.data_save_dir, 'xviewtrain_img.txt')), header=None).to_numpy()
    xview_lbl_txt = pd.read_csv(open(os.path.join(args.data_save_dir, 'xviewtrain_lbl.txt')), header=None).to_numpy()
    xview_trn_num = xview_img_txt.shape[0]

    f_img = open(os.path.join(args.data_save_dir, 'xview_{}_{}_train_img.txt'.format(syn_args.syn_display_type, syn_ratio)), 'w')
    f_lbl = open(os.path.join(args.data_save_dir, 'xview_{}_{}_train_lbl.txt'.format(syn_args.syn_display_type, syn_ratio)), 'w')

    for ix in range(xview_trn_num):
        f_img.write("%s\n" % xview_img_txt[ix, 0])
        f_lbl.write("%s\n" % xview_lbl_txt[ix, 0])

    for i in range(syn_ratio_num):
        f_img.write("%s\n" % syn_img_txt[perm_indices[i], 0])
        f_lbl.write("%s\n" % syn_lbl_txt[perm_indices[i], 0])
    f_img.close()
    f_lbl.close()

    shutil.copy(os.path.join(args.data_save_dir, 'xview_{}_{}_train_img.txt'.format(syn_args.syn_display_type, syn_ratio)),
                os.path.join(args.data_list_save_dir, 'xview_{}_{}_train_img.txt'.format(syn_args.syn_display_type, syn_ratio)))
    shutil.copy(os.path.join(args.data_save_dir, 'xview_{}_{}_train_lbl.txt'.format(syn_args.syn_display_type, syn_ratio)),
                os.path.join(args.data_list_save_dir, 'xview_{}_{}_train_lbl.txt'.format(syn_args.syn_display_type, syn_ratio)))


def collect_syn_data(data_name):
    syn_args = pps.get_syn_args()
    images_save_dir = syn_args.images_save_dir
    txt_save_dir = syn_args.txt_save_dir
    data_save_dir = syn_args.data_save_dir
    lbl_path = syn_args.annos_save_dir

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


def create_xview_syn_data(sr):
    args = pwv.get_args()
    syn_args = pps.get_syn_args()
    if sr:
        data_txt = open(os.path.join(args.data_save_dir, 'xview_{}_{}.data'.format(syn_args.syn_display_type, sr)), 'w')
        data_txt.write('train=./data_xview/{}_cls/xview_{}_{}_train_img.txt\n'.format(syn_args.class_num, syn_args.syn_display_type, sr))
        data_txt.write('train_label=./data_xview/{}_cls/xview_{}_{}_train_lbl.txt\n'.format(syn_args.class_num, syn_args.syn_display_type, sr))
    else: # sr==0
        syn_args.syn_display_type = 'syn'
        data_txt = open(os.path.join(args.data_save_dir, 'xview_{}_{}.data'.format(syn_args.syn_display_type, sr)), 'w')
        data_txt.write('train=./data_xview/{}_cls/xviewtrain_img.txt\n'.format(syn_args.class_num, syn_args.syn_display_type, sr))
        data_txt.write('train_label=./data_xview/{}_cls/xviewtrain_lbl.txt\n'.format(syn_args.class_num, syn_args.syn_display_type, sr))

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
    backup xview-plane dataset 
    clean xview-plane dataset with certain constraints
    '''
    # args = pwv.get_args()
    # catid = 0
    # whr_thres = 4  # 3
    # px_thres = 6  # 4
    # pwv.clean_backup_xview_plane_with_constraints(catid, px_thres, whr_thres)

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
    combine xview & synthetic dataset [0.25, 0.5, 0.75, 1.0]
    '''
    # syn_ratio = [0.25, 0.5, 0.75, 1.0]
    # for sr in syn_ratio:
    #     combine_xview_syn_by_ratio(sr)

    ''''
    create xview_syn_texture_0.25.data
    '''
    # syn_ratio = [0,  0.25, 0.5, 0.75, 1.0]
    # for sr in syn_ratio:
    #     create_xview_syn_data(sr)

    # sr = 0
    # create_xview_syn_data(sr)
