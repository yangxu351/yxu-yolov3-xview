import glob
import numpy as np
import math
import argparse
import os

import utils.wv_util as wv
from utils.utils_xview import coord_iou, compute_iou
from utils.xview_synthetic_util import preprocess_xview_syn_data_distribution as pps
from utils.xview_synthetic_util import process_syn_xview_background_wv_split as psx
from utils.object_score_util import get_bbox_coords_from_annos_with_object_score as gbc
import pandas as pd
from ast import literal_eval
import json
import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm
import shutil
import cv2
import seaborn as sn
import time


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

def get_annos_of_model_id(model_id=0):
    src_model_dir = args.annos_save_dir[:-1] + '_all_model/'
    des_model_modelid_dir = args.annos_save_dir[:-1] + '_m{}_only_one_with_modelid/'.format(model_id)
    if not os.path.exists(des_model_modelid_dir):
        os.mkdir(des_model_modelid_dir)
    des_model_dir = args.annos_save_dir[:-1] + '_m{}_only_one/'.format(model_id)
    if not os.path.exists(des_model_dir):
        os.mkdir(des_model_dir)

    lbl_model_txts = glob.glob(os.path.join(src_model_dir, '*.txt'))
    for lt in lbl_model_txts:
        if not pps.is_non_zero_file(lt):
            continue
        name = os.path.basename(lt)
        df_txt = pd.read_csv(lt, header=None, sep=' ')
        df_txt = df_txt[df_txt.loc[:, 5] == model_id]
        if df_txt.empty:
            continue
        df_txt.to_csv(os.path.join(des_model_modelid_dir, name), header=False, index=False, sep=' ')

        df_txt_no_model_id = df_txt.loc[:, :-1]
        df_txt_no_model_id.to_csv(os.path.join(des_model_dir, name), header=False, index=False, sep=' ')


def create_test_dataset_by_model_id(model_id):
    base_dir = args.data_save_dir
    base_val_lbl_files = pd.read_csv(os.path.join(base_dir, 'xviewval_lbl_px23whr3_seed17.txt'), header=None)
    base_val_lbl_name = [os.path.basename(f) for f in base_val_lbl_files.loc[:, 0]]

    test_lbl_files = open(os.path.join(base_dir, 'xviewtest_lbl_px23whr3_seed17_m{}_only.txt'.format(model_id)), 'w')
    test_img_files = open(os.path.join(base_dir, 'xviewtest_img_px23whr3_seed17_m{}_only.txt'.format(model_id)), 'w')
    test_lbl_with_modelid_files = open(os.path.join(base_dir, 'xviewtest_lbl_px23whr3_seed17_with_model_m{}_only.txt'.format(model_id)), 'w')

    lbl_model_with_modelid_dir = args.annos_save_dir[:-1] + '_m{}_only_one_with_modelid/'.format(model_id)
    lbl_model_dir = args.annos_save_dir[:-1] + '_m{}_only_one/'.format(model_id)
    lbl_model_txts = glob.glob(os.path.join(lbl_model_dir, '*.txt'))
    for mf in lbl_model_txts:
        m_name = os.path.basename(mf)
        img_name = m_name.replace('.txt', '.jpg')
        if m_name in base_val_lbl_name:
            test_lbl_files.write('%s\n' % mf)
            test_lbl_with_modelid_files.write('%s\n' % os.path.join(lbl_model_with_modelid_dir, m_name))
            test_img_files.write('%s\n' % os.path.join(args.images_save_dir, img_name))
    test_img_files.close()
    test_lbl_files.close()


def get_val_annos_only_model_id_labeled(model_id=0):
    src_model_dir = args.annos_save_dir[:-1] + '_all_model/'
    des_model_modelid_dir = args.annos_save_dir[:-1] + '_m{}_labeled_with_modelid/'.format(model_id)
    if not os.path.exists(des_model_modelid_dir):
        os.mkdir(des_model_modelid_dir)
    des_model_dir = args.annos_save_dir[:-1] + '_m{}_labeled/'.format(model_id)
    if not os.path.exists(des_model_dir):
        os.mkdir(des_model_dir)

    lbl_model_txts = glob.glob(os.path.join(src_model_dir, '*.txt'))
    total_targets = 0
    for lt in lbl_model_txts:
        name = os.path.basename(lt)
        if pps.is_non_zero_file(lt):
            df_txt = pd.read_csv(lt, header=None, sep=' ')
            df_txt = df_txt[df_txt.loc[:, 5] == model_id]
            if not df_txt.empty:
                df_txt_no_model_id = df_txt.loc[:, :4] # col 4 included
                total_targets += df_txt_no_model_id.shape[0]
            else:
                df_txt_no_model_id = df_txt.copy()
            df_txt.to_csv(os.path.join(des_model_modelid_dir, name), header=False, index=False, sep=' ')
            df_txt_no_model_id.to_csv(os.path.join(des_model_dir, name), header=False, index=False, sep=' ')
        else:
            shutil.copy(lt, os.path.join(des_model_modelid_dir, name))
            shutil.copy(lt, os.path.join(des_model_dir, name))
    print('total_targets ', total_targets)

def create_test_dataset_of_model_id_labeled(model_id, base_pxwhrs='px23whr3_seed17'):
    base_dir = args.data_save_dir
    base_val_lbl_files = pd.read_csv(os.path.join(base_dir, 'xviewval_lbl_px23whr3_seed17.txt'), header=None)
    base_val_lbl_name = [os.path.basename(f) for f in base_val_lbl_files.loc[:, 0]]

    test_lbl_files = open(os.path.join(base_dir, 'xviewtest_lbl_px23whr3_seed17_m{}_labeled.txt'.format(model_id)), 'w')
    test_img_files = open(os.path.join(base_dir, 'xviewtest_img_px23whr3_seed17_m{}_labeled.txt'.format(model_id)), 'w')
    test_lbl_with_modelid_files = open(os.path.join(base_dir, 'xviewtest_lbl_px23whr3_seed17_with_model_m{}_labeled.txt'.format(model_id)), 'w')

    lbl_model_with_modelid_dir = args.annos_save_dir[:-1] + '_m{}_labeled_with_modelid/'.format(model_id)
    lbl_model_dir = args.annos_save_dir[:-1] + '_m{}_labeled/'.format(model_id)
    lbl_model_txts = glob.glob(os.path.join(lbl_model_dir, '*.txt'))
    for lt in lbl_model_txts:
        lbl_name = os.path.basename(lt)
        img_name = lbl_name.replace('.txt', '.jpg')
        if lbl_name in base_val_lbl_name:
            test_lbl_files.write('%s\n' % lt)
            test_lbl_with_modelid_files.write('%s\n' % os.path.join(lbl_model_with_modelid_dir, lbl_name))
            test_img_files.write('%s\n' % os.path.join(args.images_save_dir, img_name))
    test_img_files.close()
    test_lbl_files.close()

    data_txt = open(os.path.join(base_dir, 'xviewtest_{}_with_model_m{}_labeled.data'.format(base_pxwhrs, model_id)), 'w')
    data_txt.write('classes=%s\n' % str(args.class_num))
    data_txt.write('test=./data_xview/{}_cls/{}/xviewtest_img_{}_m{}_labeled.txt\n'.format(args.class_num, base_pxwhrs, base_pxwhrs, model_id))
    data_txt.write('test_label=./data_xview/{}_cls/{}/xviewtest_lbl_{}_with_model_m{}_labeled.txt\n'.format(args.class_num, base_pxwhrs, base_pxwhrs, model_id))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(args.class_num))
    data_txt.close()


def get_annos_miss_files_empty_others_by_model_id(model_id=0):
    '''
    all annos files of model id are in only_one folder
    specified miss folder (test dataset of model_id)
    keep the miss model_id annotations, empty all other model_id files save in labeled_miss
    :param model_id:
    :return:
    '''
    base_dir = args.data_save_dir
    base_val_lbl_files = pd.read_csv(os.path.join(base_dir, 'xviewval_lbl_px23whr3_seed17.txt'), header=None)

    des_model_dir = args.annos_save_dir[:-1] + '_m{}_val_miss/'.format(model_id)
    if not os.path.exists(des_model_dir):
        os.mkdir(des_model_dir)
    des_model_modelid_dir = args.annos_save_dir[:-1] + '_m{}_val_miss_with_modelid/'.format(model_id)
    if not os.path.exists(des_model_modelid_dir):
        os.mkdir(des_model_modelid_dir)

    miss_model_dir = args.annos_save_dir[:-1] + '_m{}_miss/'.format(model_id)
    miss_files = glob.glob(os.path.join(miss_model_dir, '*.txt'))
    miss_file_names = [os.path.basename(f) for f in miss_files]
    miss_model_modelid_dir = args.annos_save_dir[:-1] + '_m{}_miss_with_modelid/'.format(model_id)

    for lv in base_val_lbl_files.loc[:, 0]:
        name = os.path.basename(lv)
        if name not in miss_file_names:
            model_txt = open(os.path.join(des_model_dir, name), 'w')
            model_modelid_txt = open(os.path.join(des_model_modelid_dir, name), 'w')
            model_txt.close()
            model_modelid_txt.close()
        else:
            shutil.copy(os.path.join(miss_model_dir, name), os.path.join(des_model_dir, name))
            shutil.copy(os.path.join(miss_model_modelid_dir, name), os.path.join(des_model_modelid_dir, name))


def create_test_dataset_of_model_id_labeled_miss(model_id, base_pxwhrs='px23whr3_seed17'):
    miss_val_dir = args.annos_save_dir[:-1] + '_m{}_val_miss/'.format(model_id)
    miss_val_lbl_files = glob.glob(os.path.join(miss_val_dir, '*.txt'))
    miss_val_lbl_name = [os.path.basename(f) for f in miss_val_lbl_files]

    base_dir = args.data_save_dir
    test_lbl_files = open(os.path.join(base_dir, 'xviewtest_lbl_px23whr3_seed17_m{}_labeled_miss.txt'.format(model_id)), 'w')
    test_img_files = open(os.path.join(base_dir, 'xviewtest_img_px23whr3_seed17_m{}_labeled_miss.txt'.format(model_id)), 'w')
    test_lbl_with_modelid_files = open(os.path.join(base_dir, 'xviewtest_lbl_px23whr3_seed17_with_model_m{}_labeled_miss.txt'.format(model_id)), 'w')

    miss_val_with_modelid_dir = args.annos_save_dir[:-1] + '_m{}_val_miss_with_modelid/'.format(model_id)

    for lm in miss_val_lbl_files:
        lbl_name = os.path.basename(lm)
        img_name = lbl_name.replace('.txt', '.jpg')
        test_lbl_files.write('%s\n' % lm)
        test_lbl_with_modelid_files.write('%s\n' % os.path.join(miss_val_with_modelid_dir, lbl_name))
        test_img_files.write('%s\n' % os.path.join(args.images_save_dir, img_name))
    test_img_files.close()
    test_lbl_files.close()

    data_txt = open(os.path.join(base_dir, 'xviewtest_{}_with_model_m{}_labeled_miss.data'.format(base_pxwhrs, model_id)), 'w')
    data_txt.write('classes=%s\n' % str(args.class_num))
    data_txt.write('test=./data_xview/{}_cls/{}/xviewtest_img_{}_m{}_labeled_miss.txt\n'.format(args.class_num, base_pxwhrs, base_pxwhrs, model_id))
    data_txt.write('test_label=./data_xview/{}_cls/{}/xviewtest_lbl_{}_with_model_m{}_labeled_miss.txt\n'.format(args.class_num, base_pxwhrs, base_pxwhrs, model_id))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(args.class_num))
    data_txt.close()


def get_txt_contain_model_id(model_id=5, copy_img=False):
    src_model_dir = args.annos_save_dir[:-1] + '_all_model/'
    des_model_dir = args.annos_save_dir[:-1] + '_m{}_all_model/'.format(model_id)
    if not os.path.exists(des_model_dir):
        os.mkdir(des_model_dir)
    if copy_img:
        src_img_dir = os.path.join(args.cat_sample_dir, 'image_with_bbox_indices/px23whr3_seed17_images_with_bbox_with_indices/')
        des_img_dir = os.path.join(args.cat_sample_dir, 'image_with_bbox_indices/px23whr3_seed17_images_with_bbox_with_indices_all_models_{}/'.format(type))
        if not os.path.exists(des_img_dir):
            os.mkdir(des_img_dir)

    base_pxwhrs = 'px23whr3_seed17'
    base_dir = args.data_save_dir
    if type != 'all':
        df_val = pd.read_csv(os.path.join(base_dir, 'xview{}_lbl_{}.txt'.format(type, base_pxwhrs)), header=None)
        lbl_model_txts = [os.path.join(src_model_dir, os.path.basename(f)) for f in df_val.loc[:, 0]]
    else:
        lbl_model_txts = glob.glob(os.path.join(src_model_dir, '*.txt'))
    for lt in lbl_model_txts:
        name = os.path.basename(lt)
        shutil.copy(lt, os.path.join(des_model_dir, name))
        if not is_non_zero_file(lt):
            continue
        if copy_img:
            img_name = name.replace('.txt', '.jpg')
            shutil.copy(os.path.join(src_img_dir, img_name),
                        os.path.join(des_img_dir, img_name))


def get_txt_contain_model_id(model_id=5, copy_img=False, type='all'):
    src_model_dir = args.annos_save_dir[:-1] + '_all_model/'
    des_model_dir = args.annos_save_dir[:-1] + '_m{}_{}_model/'.format(model_id, type)
    if not os.path.exists(des_model_dir):
        os.mkdir(des_model_dir)
    if copy_img:
        src_img_dir = os.path.join(args.cat_sample_dir, 'image_with_bbox_indices/px23whr3_seed17_images_with_bbox_with_indices/')
        des_img_dir = os.path.join(args.cat_sample_dir, 'image_with_bbox_indices/px23whr3_seed17_images_with_bbox_with_indices_m{}_{}/'.format(model_id, type))
        if not os.path.exists(des_img_dir):
            os.mkdir(des_img_dir)

    base_pxwhrs = 'px23whr3_seed17'
    base_dir = args.data_save_dir
    if type != 'all':
        df_val = pd.read_csv(os.path.join(base_dir, 'xview{}_lbl_{}.txt'.format(type, base_pxwhrs)), header=None)
        lbl_model_txts = [os.path.join(src_model_dir, os.path.basename(f)) for f in df_val.loc[:, 0]]
    else:
        lbl_model_txts = glob.glob(os.path.join(src_model_dir, '*.txt'))
    img_names = []
    for lt in lbl_model_txts:
        if not pps.is_non_zero_file(lt):
            continue
        name = os.path.basename(lt)
        df_txt = pd.read_csv(lt, header=None, sep=' ')
        contain_or_not = df_txt.loc[:, 5] == model_id # series
        contain_or_not = contain_or_not.to_numpy()
        if contain_or_not.any():
            shutil.copy(lt, os.path.join(des_model_dir, name))
            if copy_img:
                img_name = name.replace('.txt', '.jpg')
                img_names.append(img_name)
                shutil.copy(os.path.join(src_img_dir, img_name),
                            os.path.join(des_img_dir, img_name))
    print(img_names)


def label_m_val_model_with_other_label(rare_class, model_id=1, other_label=0):
    des_easy_dir = args.annos_save_dir[:-1] + '_val_m{}_to_rc{}_easy/'.format(model_id, rare_class)
    if not os.path.exists(des_easy_dir):
        os.mkdir(des_easy_dir)
    des_hard_dir = args.annos_save_dir[:-1] + '_val_m{}_to_rc{}_hard/'.format(model_id, rare_class)
    if not os.path.exists(des_hard_dir):
        os.mkdir(des_hard_dir)
    model_dir = args.annos_save_dir[:-1] + '_val_m{}_to_rc{}/'.format(model_id, rare_class)
    m_model_files = glob.glob(os.path.join(model_dir, '*.txt'))

    for f in m_model_files:
        lbl_name = os.path.basename(f)
        if not is_non_zero_file(f):
            continue
        # for easy label
        df_easy_txt = pd.read_csv(f, header=None, sep=' ')
        # df_easy_txt.loc[:, 5] = other_label
        df_easy_txt.loc[df_easy_txt.loc[:, 5] != model_id, 5] = other_label
        df_easy_txt.loc[df_easy_txt.loc[:, 5] == model_id, 5] = rare_class
        df_easy_txt.to_csv(os.path.join(des_easy_dir, lbl_name), sep=' ', header=False, index=False)
        # for hard label
        df_hard_txt = pd.read_csv(f, header=None, sep=' ')
        length = df_hard_txt.shape[0]
        for t in range(length):
            if df_hard_txt.loc[t, 5] != model_id:
                df_hard_txt = df_hard_txt.drop(t) # drop index
            else:
                df_hard_txt.loc[t, 5] = rare_class
        df_hard_txt.to_csv(os.path.join(des_hard_dir, lbl_name), sep=' ', header=False, index=False)


def get_image_list_contain_model_id(model_id):
    src_model_dir = args.annos_save_dir[:-1] + '_all_model/'
    image_list = []
    lbl_model_txts = glob.glob(os.path.join(src_model_dir, '*.txt'))
    for lt in lbl_model_txts:
        if not pps.is_non_zero_file(lt):
            continue
        name = os.path.basename(lt)
        df_txt = pd.read_csv(lt, header=None, sep=' ')
        contain_or_not = df_txt.loc[:, 5] == model_id # series
        contain_or_not = contain_or_not.to_numpy()
        if contain_or_not.any():
            image_list.append(name.replace('.txt', '.jpg'))
    return image_list


def create_model_rareclass_hard_easy_set_backup(val_m_rc_path, model_id, rare_id, non_rare_id=0, seed=199, pxwhr='px23whr3'):
    '''
    create hard easy validation dataset of model* rc*
    '''
    val_labeled_dir = os.path.join(args.data_list_save_dir, pxwhr + '_seed{}'.format(seed) + '_val_lbl_with_modelid')
    val_labeled_m_rc_easy = os.path.join(args.data_list_save_dir, pxwhr + '_seed{}'.format(seed) +
                                      '_val_lbl_m{}_rc{}_easy'.format(model_id, rare_id))
    if not os.path.exists(val_labeled_m_rc_easy):
        os.mkdir(val_labeled_m_rc_easy)

    val_labeled_m_rc_hard = os.path.join(args.data_list_save_dir, pxwhr + '_seed{}'.format(seed) +
                                      '_val_lbl_m{}_rc{}_hard'.format(model_id, rare_id))
    if not os.path.exists(val_labeled_m_rc_hard):
        os.mkdir(val_labeled_m_rc_hard)


    print('val_m_rc_path', val_m_rc_path)
    easy_m_rc_path = val_m_rc_path + '_easy'
    hard_m_rc_path = val_m_rc_path + '_hard'
    hard_m_rc_files = np.sort(glob.glob(os.path.join(hard_m_rc_path, '*.txt')))
    m_rc_names = [os.path.basename(f) for f in hard_m_rc_files]
    print('m_rc_names', m_rc_names)

    des_hard_dir = args.annos_save_dir[:-1] + \
                   '_val_m{}_rc{}_hard_seed{}'.format(model_id, rare_id, seed)
    if not os.path.exists(des_hard_dir):
        os.mkdir(des_hard_dir)
    print('des_hard_dir', des_hard_dir)

    des_easy_dir = args.annos_save_dir[:-1] + \
                   '_val_m{}_rc{}_easy_seed{}'.format(model_id, rare_id, seed)
    if not os.path.exists(des_easy_dir):
        os.mkdir(des_easy_dir)
    print('des_easy_dir', des_easy_dir)

    val_files = glob.glob(os.path.join(val_labeled_dir, '*.txt'))
    for vf in val_files:
        f_name = os.path.basename(vf)
        if is_non_zero_file(vf):
            if f_name not in m_rc_names:
                df_txt_easy = pd.read_csv(vf, header=None, sep=' ')
                df_txt_easy.loc[:, 5] = non_rare_id
            else: # easy txt have been changed for m_rc_names
                df_txt_easy = pd.read_csv(os.path.join(easy_m_rc_path, f_name), header=None, sep=' ')
            df_txt_easy.to_csv(os.path.join(des_easy_dir, f_name), header=False, index=False, sep=' ')
            shutil.copy(os.path.join(des_easy_dir, f_name), os.path.join(val_labeled_m_rc_easy, f_name))

            if f_name not in m_rc_names:
                txt_hard = open(os.path.join(des_hard_dir, f_name), 'w')
                txt_hard.close()
            else:
                df_txt_hard = pd.read_csv(os.path.join(hard_m_rc_path, f_name), header=None, sep=' ')
                df_txt_hard.to_csv(os.path.join(des_hard_dir, f_name), header=False, index=False, sep=' ')
            shutil.copy(os.path.join(des_hard_dir, f_name), os.path.join(val_labeled_m_rc_hard, f_name))

        else:
            shutil.copy(vf, os.path.join(val_labeled_m_rc_easy, f_name))
            shutil.copy(vf, os.path.join(val_labeled_m_rc_hard, f_name))

            shutil.copy(vf, os.path.join(des_easy_dir, f_name))
            shutil.copy(vf, os.path.join(des_hard_dir, f_name))


def create_test_dataset_of_m_rc(model_id, rare_id, type='hard', seed=199, pxwhrs='px23whr3_seed199'):
    val_dir = args.annos_save_dir[:-1] + '_val_m{}_rc{}_{}_seed{}'.format(model_id, rare_id, type, seed)
    print('val_dir', val_dir)
    val_lbl_files = glob.glob(os.path.join(val_dir, '*.txt'))

    base_dir = args.data_save_dir
    test_lbl_txt = open(os.path.join(base_dir, 'xviewtest_lbl_{}_m{}_rc{}_{}.txt'.format(pxwhrs, model_id, rare_id, type)), 'w')
    test_img_txt = open(os.path.join(base_dir, 'xviewtest_img_{}_m{}_rc{}_{}.txt'.format(pxwhrs, model_id, rare_id, type)), 'w')

    for lf in val_lbl_files:
        lbl_name = os.path.basename(lf)
        img_name = lbl_name.replace('.txt', '.jpg')
        test_lbl_txt.write('%s\n' % lf)
        test_img_txt.write('%s\n' % os.path.join(args.images_save_dir, img_name))
    test_img_txt.close()
    test_lbl_txt.close()

    data_txt = open(os.path.join(base_dir, 'xviewtest_{}_m{}_rc{}_{}.data'.format(pxwhrs, model_id, rare_id, type)), 'w')
    data_txt.write('classes=%s\n' % str(args.class_num))
    data_txt.write('test=./data_xview/{}_cls/{}/xviewtest_img_{}_m{}_rc{}_{}.txt\n'.format(args.class_num, pxwhrs, pxwhrs, model_id, rare_id, type))
    data_txt.write('test_label=./data_xview/{}_cls/{}/xviewtest_lbl_{}_m{}_rc{}_{}.txt\n'.format(args.class_num, pxwhrs, pxwhrs, model_id, rare_id, type))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(args.class_num))
    data_txt.close()

def convert_norm(size, box):
    '''
    https://blog.csdn.net/xuanlang39/article/details/88642010
    :param size:  w h
    :param box: y1 x1 y2 x2
    :return: xc yc w h  (relative values)
    '''
    dh = 1. / (size[1])  # h--1--y
    dw = 1. / (size[0])  # w--0--x

    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0  # (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


def val_resize_crop_by_easy_hard(scale=2, pxwhrs='px23whr3_seed17', model_id=4, rare_id=1, type='hard', px_thres=30):
    base_dir = args.data_save_dir
    img_path = os.path.join(base_dir, 'xviewtest_img_{}_m{}_rc{}_{}.txt'.format(pxwhrs, model_id, rare_id, type))
    lbl_path = os.path.join(base_dir, 'xviewtest_lbl_{}_m{}_rc{}_{}.txt'.format(pxwhrs, model_id, rare_id, type))
    df_img_files = pd.read_csv(img_path, header=None)
    df_lbl_files = pd.read_csv(lbl_path, header=None)

    save_img_dir = os.path.dirname(df_img_files.loc[0, 0]) + '_{}_upscale'.format(type)
    if not os.path.exists(save_img_dir):
        os.mkdir(save_img_dir)
    else:
        shutil.rmtree(save_img_dir)
        os.mkdir(save_img_dir)
    save_lbl_dir = os.path.dirname(df_lbl_files.loc[0, 0]) + '_upscale'
    if not os.path.exists(save_lbl_dir):
        os.mkdir(save_lbl_dir)
    else:
        shutil.rmtree(save_lbl_dir)
        os.mkdir(save_lbl_dir)

    for ix in range(df_img_files.shape[0]):
        img_file = df_img_files.loc[ix, 0]

        # print('img_file ', img_file)
        img = cv2.imread(img_file)
        h, w = img.shape[0], img.shape[1]
        img2 = cv2.resize(img, (h*scale, w*scale), interpolation=cv2.INTER_LINEAR)
        lbl_file = df_lbl_files.loc[ix, 0]

        name = os.path.basename(lbl_file)

        up_h = h*scale
        up_w = w*scale
        for i in range(scale):
            for j in range(scale):
                img_s = img2[i*h: (i+1)*h, j*w: (j+1)*w]
                cv2.imwrite(os.path.join(save_img_dir, name.split('.')[0] + '_i{}j{}.png'.format(i, j)), img_s)
                if not is_non_zero_file(lbl_file):
                    f_txt = open(os.path.join(save_lbl_dir, name.split('.')[0] + '_i{}j{}.txt'.format(i, j)), 'w')
                    f_txt.close()
        if not is_non_zero_file(lbl_file):
            continue
        if name == '2315_359.txt':
            print(lbl_file)
        lbl = pd.read_csv(lbl_file, header=None, sep=' ').to_numpy() #xc yc w h
        b0_list = []
        b1_list = []
        b2_list = []
        b3_list = []
        for ti in range(lbl.shape[0]):
            class_id = lbl[ti, 0]
            model_id = lbl[ti, -1]

            bbox = lbl[ti, 1:-1] #cid, xc, yc, w, h, mid
            # print('bbox', bbox)
        #     bbox[0] = bbox[0] * up_w
        #     bbox[1] = bbox[1] * up_h
        #     bbox[2] = bbox[2] * up_w
        #     bbox[3] = bbox[3] * up_h
        #     bbox[0] = bbox[0] - bbox[2]/2
        #     bbox[1] = bbox[1] - bbox[3]/2
        #     bbox[2] = bbox[0] + bbox[2]
        #     bbox[3] = bbox[1] + bbox[3]
        #     tl_w = 0
        #     tl_h = 0
        #     c_w = up_w/2
        #     c_h = up_h/2
        #     br_w = up_w
        #     br_h = up_h
        #     b0 = np.clip(bbox, [tl_w, tl_h, tl_w, tl_h], [c_w-1, c_h-1, c_w-1, c_h-1])
        #     b1 = np.clip(bbox, [c_w, tl_h, c_w, tl_h], [br_w-1, c_h-1, br_w-1, c_h-1])
        #     b2 = np.clip(bbox, [tl_w, c_h, tl_w, c_h], [c_w-1, br_h-1, c_w-1, br_h-1])
        #     b3 = np.clip(bbox, [c_w, c_h, c_w, c_h], [br_w-1, br_h-1, br_w-1, br_h-1])
        #     if b0[2]-b0[0] > px_thres and b0[3]-b0[1] > px_thres:
        #         b0_list.append([class_id] + convert_norm((w, h), b0) + [model_id])
        #     if b1[2]-b1[0] > px_thres and b1[3]-b1[1] > px_thres:
        #         b1[0] = b1[0] - w
        #         b1[2] = b1[2] - w
        #         b1_list.append([class_id] + convert_norm((w, h), b1) + [model_id])
        #     if b2[2]-b2[0] > px_thres and b2[3]-b2[1] > px_thres:
        #         b2[1] = b2[1] - h
        #         b2[3] = b2[3] - h
        #         b2_list.append([class_id] + convert_norm((w, h), b2) + [model_id])
        #     if b3[2]-b3[0] > px_thres and b3[3]-b3[1] > px_thres:
        #         b3 = [bx - w for bx in b3]
        #         b3_list.append([class_id] + convert_norm((w, h), b3) + [model_id])  # cid xc, yc, w, h mid
        # if len(b0_list):
        #     f_txt = open(os.path.join(save_lbl_dir, name.split('.')[0] + '_i0j0.txt'), 'w')
        #     for i0 in b0_list: # xc, yc, w, h
        #         f_txt.write( "%s %s %s %s %s %s\n" % (np.int(i0[0]), i0[2], i0[1], i0[4], i0[3], np.int(i0[5])))
        #     f_txt.close()
        # if len(b1_list):
        #     f_txt = open(os.path.join(save_lbl_dir, name.split('.')[0] + '_i0j1.txt'), 'w')
        #     for i1 in b1_list:
        #         # print('i1', i1)
        #         f_txt.write( "%s %s %s %s %s %s\n" % (np.int(i1[0]), i1[2], i1[1], i1[4], i1[3], np.int(i1[5])))
        #     f_txt.close()
        # if len(b2_list):
        #     f_txt = open(os.path.join(save_lbl_dir, name.split('.')[0] + '_i1j0.txt'), 'w')
        #     for i2 in b2_list:
        #         f_txt.write( "%s %s %s %s %s %s\n" % (np.int(i2[0]), i2[2], i2[1], i2[4], i2[3], np.int(i2[5])))
        #     f_txt.close()
        # if len(b3_list):
        #     f_txt = open(os.path.join(save_lbl_dir, name.split('.')[0] + '_i1j1.txt'), 'w')
        #     for i3 in b3_list:
        #         f_txt.write( "%s %s %s %s %s %s\n" % (np.int(i3[0]), i3[2], i3[1], i3[4], i3[3], np.int(i3[5])))
        #     f_txt.close()

            #cid, xc, yc, w, h, mid
            # bbox[0] = bbox[0] * up_h
            # bbox[1] = bbox[1] * up_w
            # bbox[2] = bbox[2] * up_h
            # bbox[3] = bbox[3] * up_w
            # bbox[0] = bbox[0] - bbox[2]/2
            # bbox[1] = bbox[1] - bbox[3]/2
            # bbox[2] = bbox[0] + bbox[2]
            # bbox[3] = bbox[1] + bbox[3]
            bbox[0] = bbox[0] * up_w
            bbox[1] = bbox[1] * up_h
            bbox[2] = bbox[2] * up_w
            bbox[3] = bbox[3] * up_h
            bbox[0] = bbox[0] - bbox[2]/2
            bbox[1] = bbox[1] - bbox[3]/2
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]
            tl_w = 0
            tl_h = 0
            c_w = up_w/2
            c_h = up_h/2
            br_w = up_w
            br_h = up_h
            b0 = np.clip(bbox, [tl_w, tl_h, tl_w, tl_h], [c_w-1, c_h-1, c_w-1, c_h-1])
            b1 = np.clip(bbox, [c_w, tl_h, c_w, tl_h], [br_w-1, c_h-1, br_w-1, c_h-1])
            b2 = np.clip(bbox, [tl_w, c_h, tl_w, c_h], [c_w-1, br_h-1, c_w-1, br_h-1])
            b3 = np.clip(bbox, [c_w, c_h, c_w, c_h], [br_w-1, br_h-1, br_w-1, br_h-1])
            if b0[2]-b0[0] > px_thres and b0[3]-b0[1] > px_thres:
                b0_list.append([class_id] + convert_norm((w, h), b0) + [model_id])
            if b1[2]-b1[0] > px_thres and b1[3]-b1[1] > px_thres:
                b1[0] = b1[0] - w
                b1[2] = b1[2] - w
                b1_list.append([class_id] + convert_norm((w, h), b1) + [model_id])
            if b2[2]-b2[0] > px_thres and b2[3]-b2[1] > px_thres:
                b2[1] = b2[1] - h
                b2[3] = b2[3] - h
                b2_list.append([class_id] + convert_norm((w, h), b2) + [model_id])
            if b3[2]-b3[0] > px_thres and b3[3]-b3[1] > px_thres:
                b3 = [bx - w for bx in b3]
                b3_list.append([class_id] + convert_norm((w, h), b3) + [model_id]) #cid, xc, yc, w, h, mid
        f_txt = open(os.path.join(save_lbl_dir, name.split('.')[0] + '_i0j0.txt'), 'w')
        for i0 in b0_list:
            f_txt.write( "%s %s %s %s %s %s\n" % (np.int(i0[0]), i0[1], i0[2], i0[3], i0[4], np.int(i0[5])))
        f_txt.close()
        f_txt = open(os.path.join(save_lbl_dir, name.split('.')[0] + '_i0j1.txt'), 'w')
        for i1 in b1_list:
            # print('i1', i1)
            f_txt.write( "%s %s %s %s %s %s\n" % (np.int(i1[0]), i1[1], i1[2], i1[3], i1[4], np.int(i1[5])))
        f_txt.close()
        f_txt = open(os.path.join(save_lbl_dir, name.split('.')[0] + '_i1j0.txt'), 'w')
        for i2 in b2_list:
            f_txt.write( "%s %s %s %s %s %s\n" % (np.int(i2[0]), i2[1], i2[2], i2[3], i2[4], np.int(i2[5])))
        f_txt.close()
        f_txt = open(os.path.join(save_lbl_dir, name.split('.')[0] + '_i1j1.txt'), 'w')
        for i3 in b3_list: #cid, xc, yc, w, h, mid
            f_txt.write( "%s %s %s %s %s %s\n" % (np.int(i3[0]), i3[1], i3[2], i3[3], i3[4], np.int(i3[5])))
        f_txt.close()



def create_upsample_test_dataset_of_m_rc(model_id, rare_id, type='hard', seed=17, pxwhrs='px23whr3_seed17'):
    val_dir = args.annos_save_dir[:-1] + '_val_m{}_rc{}_{}_seed{}_upscale'.format(model_id, rare_id, type, seed)
    print('val_dir', val_dir)
    val_lbl_files = glob.glob(os.path.join(val_dir, '*.txt'))

    base_dir = args.data_save_dir
    test_lbl_txt = open(os.path.join(base_dir, 'xviewtest_lbl_{}_upscale_m{}_rc{}_{}.txt'.format(pxwhrs, model_id, rare_id, type)), 'w')
    test_img_txt = open(os.path.join(base_dir, 'xviewtest_img_{}_upscale_m{}_rc{}_{}.txt'.format(pxwhrs, model_id, rare_id, type)), 'w')

    for lf in val_lbl_files:
        lbl_name = os.path.basename(lf)
        img_name = lbl_name.replace('.txt', '.png')
        test_lbl_txt.write('%s\n' % lf)
        test_img_txt.write('%s\n' % os.path.join(args.images_save_dir[:-1] + '_{}_upscale'.format(type), img_name))
    test_img_txt.close()
    test_lbl_txt.close()

    data_txt = open(os.path.join(base_dir, 'xviewtest_{}_upscale_m{}_rc{}_{}.data'.format(pxwhrs, model_id, rare_id, type)), 'w')
    data_txt.write('classes=%s\n' % str(args.class_num))
    data_txt.write('test=./data_xview/{}_cls/{}/xviewtest_img_{}_upscale_m{}_rc{}_{}.txt\n'.format(args.class_num, pxwhrs, pxwhrs, model_id, rare_id, type))
    data_txt.write('test_label=./data_xview/{}_cls/{}/xviewtest_lbl_{}_upscale_m{}_rc{}_{}.txt\n'.format(args.class_num, pxwhrs, pxwhrs, model_id, rare_id, type))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(args.class_num))
    data_txt.close()

def get_args(px_thres=None, whr_thres=None, seed=17):
    parser = argparse.ArgumentParser()

    parser.add_argument("--xview_yolo_dir", type=str, help="dir to xViewYOLO",
                        default='/media/lab/Yang/data/xView_YOLO/')

    parser.add_argument("--images_save_dir", type=str, help="to save chip trn val images files",
                        default='/media/lab/Yang/data/xView_YOLO/images/')

    parser.add_argument("--txt_save_dir", type=str, help="to save  related label files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')

    parser.add_argument("--data_list_save_dir", type=str, help="to save selected trn val images and labels",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_cls/data_list/')

    parser.add_argument("--data_save_dir", type=str, help="to save data files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_cls/')
    parser.add_argument("--cat_sample_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/')
    parser.add_argument("--annos_save_dir", type=str, help="to save txt annotation files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')
    parser.add_argument("--font3", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 10}")
    parser.add_argument("-ft2", "--font2", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 13}")
    parser.add_argument("--class_num", type=int, default=1, help="Number of Total Categories")  # 60  6 1
    parser.add_argument("--input_size", type=int, default=608, help="Number of Total Categories")  # 300 416

    args = parser.parse_args()
    #fixme ----------==--------change
    args.images_save_dir = args.images_save_dir + '{}_{}cls/'.format(args.input_size, args.class_num)
    if px_thres:
        args.data_save_dir = args.data_save_dir.format(args.class_num) + 'px{}whr{}_seed{}/'.format(px_thres, whr_thres, seed)
        args.annos_save_dir = args.annos_save_dir + '{}/{}_cls_xcycwh_px{}whr{}/'.format(args.input_size, args.class_num, px_thres, whr_thres)
    else:
        args.data_save_dir = args.data_save_dir.format(args.class_num)
        args.annos_save_dir = args.annos_save_dir + '{}/{}_cls_xcycwh/'.format(args.input_size, args.class_num)
    args.txt_save_dir = args.txt_save_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)
    args.cat_sample_dir = args.cat_sample_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)

    args.data_list_save_dir = args.data_list_save_dir.format(args.input_size, args.class_num)
    if not os.path.exists(args.txt_save_dir):
        os.makedirs(args.txt_save_dir)

    if not os.path.exists(args.annos_save_dir):
        os.makedirs(args.annos_save_dir)

    if not os.path.exists(args.images_save_dir):
        os.makedirs(args.images_save_dir)

    if not os.path.exists(args.cat_sample_dir):
        os.makedirs(args.cat_sample_dir)

    return args


def get_rotated_point(x,y,angle):
    '''
    https://blog.csdn.net/weixin_44135282/article/details/89003793
    '''
    # (h, w) = image.shape[:2]
    # # 将图像中心设为旋转中心
    w, h = 1, 1
    (cX, cY) = (0.5, 0.5)

    #假设图像的宽度x高度为col*row, 图像中某个像素P(x1, y1)，绕某个像素点Q(x2, y2)
    #旋转θ角度后, 则该像素点的新坐标位置为(x, y)，其计算公式为：

    x = x
    y = h - y
    cX = cX
    cY = h - cY
    new_x = (x - cX) * math.cos(math.pi / 180.0 * angle) - (y - cY) * math.sin(math.pi / 180.0 * angle) + cX
    new_y = (x - cX) * math.sin(math.pi / 180.0 * angle) + (y - cY) * math.cos(math.pi / 180.0 * angle) + cY
    new_x = new_x
    new_y = h - new_y
    # return round(new_x), round(new_y) #四舍五入取整
    return new_x, new_y

def get_flipped_point(x, y, flip='tb'):
    w, h = 1, 1
    if flip == 'tb':
        new_y = h - y
        new_x = x
    elif flip == 'lr':
        new_x = w - x
        new_y = y
    return new_x, new_y


if __name__ == '__main__':
    whr_thres = 3
    px_thres = 23
    seed = 17
    args = get_args(px_thres, whr_thres, seed)

    '''
    change end model id 
    '''
    # px_thres = 23
    # whr_thres = 3
    # end_mid = 6
    # dc.change_end_model_id(px_thres,whr_thres, end_mid)

    '''
    get model 0 txt only
    only one model label
    '''
    # # model_id = 0
    # model_id = 4
    # # model_id = 1
    # get_annos_of_model_id(model_id)

    '''
    create test dataset contain only one model
    '''
    # model_id = 1
    # model_id = 4
    # create_test_dataset_by_model_id(model_id)

    '''
    create *.data for only one model
    '''
    # base_cmt = 'px23whr3_seed17'
    # # model_id = 1
    # model_id = 4
    # psx.create_xview_base_data_for_onemodel_only(model_id, base_cmt)

    '''
    get lbl txt and images  of val or train
    '''
    # get_trn_val_txt_contain_all_models(type='val', copy_img=True)

    '''
    get txt which contains model_id 
    '''
    # model_id = 4
    # model_id = 1
    # model_id = 5
    # type = 'val'
    # get_txt_contain_model_id(model_id, copy_img=True, type=type)

    '''
    manually backup _m*_val_model as _m*_miss_val_model  
    manually select rc2 from m1_val_model 
    except rc2 all others of model1 labeled as 0
    '''
    # model_id = 1
    # rare_class = 2
    # model_id = 5
    # rare_class = 3
    # other_label = 0
    # label_m_val_model_with_other_label(rare_class, model_id, other_label)
    '''
    get txt which contains model_id == 5
    '''
    # model_id = 4
    # get_txt_contain_model_id(model_id, copy_img=True)

    '''
    get image_list that contain model_id
    '''
    # model_id = 4
    # model_id = 3
    # model_id = 2
    # image_list = get_image_list_contain_model_id(model_id)
    # print(image_list)

    '''                                                                                                 
    create the validation set of model * that used for zero-learning (easy) and zero-learning (hard)
    hard------> val set : only rc* labeled, others empty
    easy------> val set : rc* labeled as rc, others labeled as non-rc(0)    
    seed = 17                                                                                           
    seed = 199                                                                                          
    all models that are not belong to the rare object will be labeled as 0  
    '''
    # seed = 17
    # # seed = 199
    # px_thres = 23
    # whr_thres = 3
    # args = get_args(px_thres, whr_thres, seed)
    # pxwhr = 'px{}whr{}'.format(px_thres, whr_thres)
    # non_rare_id = 0
    # # model_id = 4
    # # rare_id = 1
    # # val_m_rc_path = args.annos_save_dir[:-1] + '_m{}_rc{}'.format(model_id, rare_id)
    # # model_id = 1
    # # rare_id = 2
    # model_id = 5
    # rare_id = 3
    # val_m_rc_path = args.annos_save_dir[:-1] + '_val_m{}_to_rc{}'.format(model_id, rare_id)
    # create_model_rareclass_hard_easy_set_backup(val_m_rc_path, model_id, rare_id, non_rare_id, seed, pxwhr)

    '''                                                                                                 
    create *.data for zero-learning (easy) and zero-learning (hard)
    hard------> val set : only rc* labeled, others empty
    easy------> val set : rc* labeled as rc, others labeled as non-rc(0)    
    seed = 17                                                                                           
    seed = 199                                                                                          
    '''
    # seed = 17
    # # seed = 199
    # px_thres = 23
    # whr_thres = 3
    # args = get_args(px_thres, whr_thres, seed)
    # pxwhrs = 'px{}whr{}_seed{}'.format(px_thres, whr_thres, seed)
    # # model_id = 4
    # # rare_id = 1
    # # model_id = 1
    # # rare_id = 2
    # model_id = 5
    # rare_id = 3
    # non_rare_id = 0
    # types = ['hard', 'easy']
    # for type in types:
    #     create_test_dataset_of_m_rc(model_id, rare_id, type, seed, pxwhrs)


    # import collections
    # print([item for item, count in collections.Counter(trn_names).items() if count > 1])



    '''
    get all validation txt but only model_id labeled
    others are empty
    '''
    # # model_id = 0
    # model_id = 4
    # model_id = 1
    # get_val_annos_only_model_id_labeled(model_id)


    '''
    get all validation txt but only specified miss model_id labeled
    all others are empty except the miss ones
    '''
    # model_id = 4
    # get_annos_miss_files_empty_others_by_model_id(model_id)

    '''
    get all validation txt but only specified miss model_id labeled
    base on val miss list
    others are empty
    '''
    # model_id = 4
    # # model_id = 1
    # create_test_dataset_of_model_id_labeled_miss(model_id)

    '''
    create val dataset of all annos but only model_id labeled
    others are empty
    '''
    # model_id = 1
    # model_id = 4
    # create_test_dataset_of_model_id_labeled(model_id)


    '''
    resize validation images and labels
    crop
    create new val*_upscale*.data
    '''
    scale=2
    px_thres = 30
    pxwhrs='px23whr3_seed17'
    model_id=4
    rare_id=1
    # type='hard'
    type='easy'
    val_resize_crop_by_easy_hard(scale, pxwhrs, model_id, rare_id, type, px_thres)
    create_upsample_test_dataset_of_m_rc(model_id, rare_id, type, seed=17, pxwhrs='px23whr3_seed17')
    '''
    check annotation
     plot images with bbox
    '''
    # type='hard'
    type='easy'
    save_dir = os.path.join(args.cat_sample_dir, 'image_with_bbox/2315_{}_upscale/'.format(type))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    lbl_dir = args.annos_save_dir[:-1] + '_val_m4_rc1_{}_seed17_upscale/'.format(type)
    img_dir = args.images_save_dir[:-1] + '_{}_upscale/'.format(type)
    img_list = glob.glob(os.path.join(img_dir, '2315_*.png'))
    for f in img_list:
        print('f ', f)
        name = os.path.basename(f)
        lbl_file = os.path.join(lbl_dir, name.replace('.png', '.txt'))
        gbc.plot_img_with_bbx(f, lbl_file, save_dir)

    '''
    flip and rotate images 
    '''
    # import PIL
    # from PIL import Image
    # img = Image.open('/media/lab/Yang/data/xView_YOLO/images/608_1cls_of_2315_359/2315_359.jpg')
    # out_lr_flip = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    # out_lr_flip.save('/media/lab/Yang/data/xView_YOLO/images/608_1cls_of_2315_359/2315_359_fl.jpg')
    # out_tb_flip = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    # out_tb_flip.save('/media/lab/Yang/data/xView_YOLO/images/608_1cls_of_2315_359/2315_359_tb.jpg')
    # out_rt_90 = img.transpose(PIL.Image.ROTATE_90)
    # out_rt_90.save('/media/lab/Yang/data/xView_YOLO/images/608_1cls_of_2315_359/2315_359_rt90.jpg')
    # out_rt_180 = img.transpose(PIL.Image.ROTATE_180)
    # out_rt_180.save('/media/lab/Yang/data/xView_YOLO/images/608_1cls_of_2315_359/2315_359_rt180.jpg')
    # out_rt_270 = img.transpose(PIL.Image.ROTATE_270)
    # out_rt_270.save('/media/lab/Yang/data/xView_YOLO/images/608_1cls_of_2315_359/2315_359_rt270.jpg')

    '''
    flip and rotate coordinates of bbox 
    '''
    # from utils.object_score_util import get_bbox_coords_from_annos_with_object_score as gbc
    # lbl_dir = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px23whr3_val_m4_rc1_2315_259/'
    # img_dir = '/media/lab/Yang/data/xView_YOLO/images/608_1cls_of_2315_359/'
    # save_dir = '/media/lab/Yang/data/xView_YOLO/cat_samples/608/1_cls/image_with_bbox/2315_359_aug/'
    #
    # # angle = 270 # 180 # 90
    # # lbl_file = os.path.join(lbl_dir, '2315_359_rt{}.txt'.format(angle))
    # flip = 'tb' # lr
    # lbl_file = os.path.join(lbl_dir, '2315_359_{}.txt'.format(flip))
    # df_lf = pd.read_csv(lbl_file, header=None, sep=' ')
    # for i in range(df_lf.shape[0]):
    #     # df_lf.loc[i, 1], df_lf.loc[i, 2] = get_rotated_point(df_lf.loc[i, 1], df_lf.loc[i, 2], angle)
    #     df_lf.loc[i, 1], df_lf.loc[i, 2] = get_flipped_point(df_lf.loc[i, 1], df_lf.loc[i, 2], flip)
    # df_lf.to_csv(lbl_file, header=False, index=False, sep=' ')
    # name = os.path.basename(lbl_file)
    # print('name', name)
    # img_name = name.replace('.txt', '.jpg')
    # img_file = os.path.join(img_dir, img_name)
    # gbc.plot_img_with_bbx(img_file, lbl_file, save_path=save_dir)


