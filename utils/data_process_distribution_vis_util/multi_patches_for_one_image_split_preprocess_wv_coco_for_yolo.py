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
from utils.object_score_util import get_bbox_coords_from_annos_with_object_score as gbc
import time


def get_multi_chips_and_txt_geojson_2_json_of_tif_name(tif_name):
    '''
    :return:
    catid_images_name_maps
    catid_tifs_name_maps
    copy raw tif to septerate tif folder
    '''
    coords, chips, classes, features_ids = wv.get_labels(args.json_filepath, args.class_num)
    img_file =os.path.join(args.image_folder, tif_name)
    arr = wv.get_image(img_file)
    res = (args.input_size, args.input_size)

    img_save_dir = args.images_save_dir[:-1] + '_of_{}'.format(tif_name.split('.')[0])
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)

    ims, img_names, box, classes_final, box_ids = wv.chip_image_with_sliding_widow(arr, coords[chips == tif_name],
                                                                    classes[chips == tif_name],
                                                                    features_ids[chips == tif_name], res,
                                                                    tif_name.split('.')[0], img_save_dir)

    txt_norm_dir = args.annos_save_dir[:-1] + '_of_{}'.format(tif_name.split('.')[0])
    if not os.path.exists(txt_norm_dir):
        os.makedirs(txt_norm_dir)

    txt_save_dir = args.txt_save_dir[:-1] + '_of_{}'.format(tif_name.split('.')[0])
    if not os.path.exists(txt_save_dir):
        os.mkdir(txt_save_dir)

    _img_num = 0
    img_num_list = []
    image_info_list = []
    annotation_list = []
    image_names_list = []
    ks = [k for k in ims.keys()]
    # print('ks ', len(ks))
    for k in ks:
        file_name = img_names[k]
        file_name_pref = file_name.split('.')[0]
        image_names_list.append(file_name)
        ana_txt_name = file_name.split(".")[0] + ".txt"
        f_txt = open(os.path.join(txt_norm_dir, ana_txt_name), 'w')
        img = wv.get_image(os.path.join(img_save_dir, file_name))
        image_info = {
            "id": _img_num,
            "file_name": file_name,
            "height": img.shape[0],
            "width": img.shape[1],
            "date_captured": datetime.datetime.utcnow().isoformat(' '),
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        }
        image_info_list.append(image_info)

        for d in range(box_ids[k].shape[0]):
            # create annotation_info
            bbx = box[k][d]
            annotation_info = {
                "id": box_ids[k][d],
                "image_id": _img_num,
                # "image_name": img_name, #fixme: there isn't 'image_name'
                "category_id": np.int(classes_final[k][d]),
                "iscrowd": 0,
                "area": (bbx[2] - bbx[0] + 1) * (bbx[3] - bbx[1] + 1),
                "bbox": [bbx[0], bbx[1], bbx[2] - bbx[0], bbx[3] - bbx[1]],  # (w1, h1, w, h)
                # "segmentation": [],
            }
            annotation_list.append(annotation_info)

            bbx = [np.int(b) for b in box[k][d]]
            cvt_bbx = pwv.convert_norm(res, bbx)
            f_txt.write(
                "%s %s %s %s %s\n" % (np.int(classes_final[k][d]), cvt_bbx[0], cvt_bbx[1], cvt_bbx[2], cvt_bbx[3]))
        img_num_list.append(_img_num)
        _img_num += 1
        f_txt.close()
    print('_img_num', _img_num)

    trn_instance = {'info': '{} {} cls chips 608 yx185 created {}'.format(file_name, args.class_num, time.strftime('%Y-%m-%d_%H.%M', time.localtime())),
                    'license': 'license', 'images': image_info_list,
                    'annotations': annotation_list, 'categories': wv.get_all_categories(args.class_num)}
    json_file = os.path.join(txt_save_dir,
                             'xview_{}_{}_{}cls_xtlytlwh.json'.format(file_name_pref, args.input_size, args.class_num))  # topleft
    json.dump(trn_instance, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=pwv.MyEncoder)



def get_img_txt_for_multi_chips(tif_name):
    txt_norm_dir = args.annos_save_dir[:-1] + '_of_{}'.format(tif_name.split('.')[0])
    image_rc_val_dir = os.path.join(args.images_save_dir[:-1] + '_of_{}'.format(tif_name.split('.')[0]), '{}_rc_val'.format(tif_name.split('.')[0]))
    image_list = glob.glob(os.path.join(image_rc_val_dir, '*.jpg'))
    txt_rc_val_dest_dir = args.annos_save_dir[:-1] + '_rc_val_multi'
    if not os.path.exists(txt_rc_val_dest_dir):
        os.mkdir(txt_rc_val_dest_dir)
    multi_img_dir = args.images_save_dir[:-1] + '_rc_val_multi_crops'
    if not os.path.exists(multi_img_dir):
        os.mkdir(multi_img_dir)
    for f in image_list:
        shutil.copy(f, os.path.join(multi_img_dir, os.path.basename(f)))
        lbl_name = os.path.basename(f).replace('.jpg', '.txt')
        shutil.copy(os.path.join(txt_norm_dir, lbl_name),
                    os.path.join(txt_rc_val_dest_dir, lbl_name))


def draw_bbox_with_indices(tif_name):
    # txt_rc_val_dir = args.annos_save_dir[:-1] + '_rc_val_multi_with_modelid'
    txt_rc_val_dir = args.annos_save_dir[:-1] + '_rc_val_new_ori_multi_modelid'
    # txt_rc_val_dir = args.annos_save_dir[:-1] + '_rc_val_multi'
    image_rc_val_dir = os.path.join(args.images_save_dir[:-1] + '_of_{}'.format(tif_name.split('.')[0]), '{}_rc_val'.format(tif_name.split('.')[0]))
    image_list = glob.glob(os.path.join(image_rc_val_dir, '*.jpg'))
    bbox_folder_name = 'val_rc_images_{}'.format(tif_name.split('.tif')[0])
    img_with_bbx_dir = os.path.join(args.cat_sample_dir, 'image_with_bbox_indices_val', bbox_folder_name) #
    if not os.path.exists(img_with_bbx_dir):
        os.makedirs(img_with_bbx_dir)
    for f in image_list:
        print(f)
        print('img_with_bbx_dir', img_with_bbx_dir)
        lbl = os.path.join(txt_rc_val_dir, os.path.basename(f).replace('.jpg', '.txt'))
        print('lbl', lbl)
        gbc.plot_img_with_bbx(f, lbl, img_with_bbx_dir, label_index=True)


def combine_ori_multi_img_lbl(px_thres=23, whr_thres=3):
    ori_val_img_dir = args.images_save_dir[:-1] + '_rc_val'
    ori_val_img_files = glob.glob(os.path.join(ori_val_img_dir, '*.jpg'))

    multi_img_dir = args.images_save_dir[:-1] + '_rc_val_multi_crops'
    multi_img_files = glob.glob(os.path.join(multi_img_dir, '*.jpg'))

    new_val_img_list = ori_val_img_files + multi_img_files
    new_val_img_dir = args.images_save_dir[:-1] + '_rc_val_new_ori_multi'
    if not os.path.exists(new_val_img_dir):
        os.mkdir(new_val_img_dir)
    for f in new_val_img_list:
        name = os.path.basename(f)
        shutil.copy(f, os.path.join(new_val_img_dir, name))

    nrc_rc_ori_lbl_model_dir = args.annos_save_dir[:-1] + '_with_rcid'
    ori_lbl_model_dir = args.annos_save_dir[:-1] + '_rc_val_ori_rcid'
    if not os.path.exists(ori_lbl_model_dir):
        os.mkdir(ori_lbl_model_dir)
        for vi in ori_val_img_files:
            lbl_name = os.path.basename(vi).replace('.jpg', '.txt')
            shutil.copy(os.path.join(nrc_rc_ori_lbl_model_dir, lbl_name), os.path.join(ori_lbl_model_dir, lbl_name))
    multi_lbl_model_dir = args.annos_save_dir[:-1] + '_rc_val_multi_with_rcid'
    new_val_lbl_model_dir = args.annos_save_dir[:-1] + '_rc_val_new_ori_multi_rcid'
    if not os.path.exists(new_val_lbl_model_dir):
        os.mkdir(new_val_lbl_model_dir)

    multi_lbl_list = glob.glob(os.path.join(multi_lbl_model_dir, '*.txt'))
    for lbl in multi_lbl_list:
        df_lbl = pd.read_csv(lbl, header=None, sep=' ')
        df_filter = df_lbl.copy()
        num = df_lbl.shape[0]
        for ix in range(num):
            df_filter.loc[ix, 1:] = df_filter.loc[ix, 1:] * args.input_size
            w = df_filter.loc[ix, 3]
            h = df_filter.loc[ix, 4]
            lt_w = df_filter.loc[ix, 1] - w/2
            lt_h = df_filter.loc[ix, 2] - w/2
            rb_w = df_filter.loc[ix, 1] + w/2
            rb_h = df_filter.loc[ix, 2] + w/2
            # edge = args.input_size - 1
            if (lt_h<=0 or lt_w<=0 or rb_h<=0 or rb_w<=0) and (w/h>=whr_thres or h/w>=whr_thres or w<=px_thres or h<=px_thres):
                df_filter.drop(ix)
                df_lbl = df_lbl.drop(ix)
            elif (lt_h>=args.input_size or lt_w>=args.input_size or rb_h>=args.input_size or rb_w>=args.input_size) and (w/h>=whr_thres or h/w>=whr_thres or w<=px_thres or h<=px_thres):
                df_filter.drop(ix)
                df_lbl = df_lbl.drop(ix)
        df_lbl.to_csv(lbl, header=False, index=False, sep=' ')
    new_lbl_files_list = glob.glob(os.path.join(ori_lbl_model_dir, '*.txt')) + glob.glob(os.path.join(multi_lbl_model_dir, '*.txt'))
    for f in new_lbl_files_list:
        name = os.path.basename(f)
        shutil.copy(f, os.path.join(new_val_lbl_model_dir, name))


def split_trn_val_with_aug_rc_nrcbkg_step_by_step(data_name='xview', comments='', seed=17):
    '''
    first step: split data contains aircrafts but no rc images
    second step: split data contains no aircrafts (bkg images)
    third step: split RC images train:val targets ~ 1:1 !!!! manully split
    fourth step: combine agumented RC
    ###################### important!!!!! the difference between '*.txt' and '_*.txt'
    ######################  set(l) will change the order of list
                           list(dict.fromkeys(l)) doesn't change the order of list
    '''
    data_save_dir = args.data_save_dir
    if comments:
        txt_save_dir = args.data_list_save_dir + comments[1:] # + '_bh'+ '/'
        if not os.path.exists(txt_save_dir):
            os.makedirs(txt_save_dir)
        data_save_dir = os.path.join(data_save_dir, comments[1:])
        if not os.path.exists(data_save_dir):
            os.makedirs(data_save_dir)
    else:
        txt_save_dir = args.data_list_save_dir

    lbl_model_path = args.annos_save_dir[:-1] + '_with_rcid'
    print('lbl_path', lbl_model_path)
    bkg_lbl_dir = args.annos_save_dir[:-1] + '_bkg'

    images_save_dir = args.images_save_dir
    trn_rc_img_dir = args.images_save_dir[:-1] + '_rc_train'
    val_rc_img_dir = args.images_save_dir[:-1] + '_rc_val'
    rc_img_dir = args.images_save_dir[:-1] + '_rc'
    bkg_img_dir = args.images_save_dir[:-1] + '_noairplane_bkg_chips'

    ##### rare classes
    all_rc_imgs = glob.glob(os.path.join(trn_rc_img_dir, '*.jpg')) + glob.glob(os.path.join(val_rc_img_dir, '*.jpg'))
    all_rc_img_names = [os.path.basename(f) for f in all_rc_imgs]
    all_rc_lbl_names = [f.replace('.jpg', '.txt') for f in all_rc_img_names]
    print('all_rc_img_names', len(all_rc_img_names))

    # print('trn_rc_lbl_files', trn_rc_lbl_files)
    trn_rc_img_files = [f for f in glob.glob(os.path.join(trn_rc_img_dir, '*.jpg'))]
    val_rc_img_files = [f for f in glob.glob(os.path.join(val_rc_img_dir, '*.jpg'))]
    print('trn_rc_img_files', len(trn_rc_img_files))
    print('val_rc_img_files', len(val_rc_img_files))
    trn_rc_lbl_files = [os.path.join(lbl_model_path, os.path.basename(f).replace('.jpg', '.txt')) for f in trn_rc_img_files]
    val_rc_lbl_files = [os.path.join(lbl_model_path, os.path.basename(f).replace('.jpg', '.txt')) for f in val_rc_img_files]

    val_aug_rc_img_dir =  args.images_save_dir[:-1] + '_rc_val_new_ori_multi_aug'
    val_aug_rc_lbl_dir = args.annos_save_dir[:-1] + '_rc_val_new_ori_multi_rcid_aug'

    val_aug_rc_img_files = glob.glob(os.path.join(val_aug_rc_img_dir, '*.jpg'))
    val_aug_rc_lbl_files = [os.path.join(val_aug_rc_lbl_dir, os.path.basename(f).replace('.jpg', '.txt')) for f in val_aug_rc_img_files]

    # print('trn_rc_lbl_files', trn_rc_lbl_files)
    ##### images that contain aircrafts
    airplane_lbl_files = [f for f in glob.glob(os.path.join(lbl_model_path, '*.txt')) if pps.is_non_zero_file(f)]
    airplane_lbl_files.sort()
    num_air_files = len(airplane_lbl_files)
    print('totally 441 images, num_air_files', num_air_files)

    ##### images that contain no aircrafts (drop out by rules)
    airplane_ept_lbl_files = [os.path.join(lbl_model_path, os.path.basename(f)) for f in glob.glob(os.path.join(lbl_model_path, '*.txt')) if not pps.is_non_zero_file(f)]
    airplane_ept_img_files = [os.path.join(images_save_dir, os.path.basename(f).replace('.txt', '.jpg')) for f in airplane_ept_lbl_files]
    print('airplane_ept_img_files', len(airplane_ept_img_files))

    ##### images that contain no aircrafts-- BKG
    bkg_lbl_files = glob.glob(os.path.join(bkg_lbl_dir, '*.txt'))
    bkg_lbl_files.sort()
    print('bkg_lbl_files', len(bkg_lbl_files))
    bkg_img_files = [os.path.join(bkg_img_dir, os.path.basename(f).replace('.txt', '.jpg')) for f in bkg_lbl_files]
    bkg_lbl_files = bkg_lbl_files + airplane_ept_lbl_files
    bkg_img_files = bkg_img_files + airplane_ept_img_files

    np.random.seed(seed)
    nrc_lbl_files = [f for f in airplane_lbl_files if os.path.basename(f) not in all_rc_lbl_names]
    nrc_img_files = [os.path.join(images_save_dir, os.path.basename(f).replace('.txt', '.jpg')) for f in nrc_lbl_files]
    print('len nrc img, len nrc lbl',len(nrc_img_files), len(nrc_lbl_files))

    nrc_ixes = np.random.permutation(len(nrc_lbl_files))
    nrc_val_num = int(len(nrc_lbl_files)*args.val_percent)
    val_nrc_lbl_files = [nrc_lbl_files[i] for i in nrc_ixes[:nrc_val_num]]
    val_nrc_img_files = [nrc_img_files[i] for i in nrc_ixes[:nrc_val_num]]
    trn_nrc_lbl_files = [nrc_lbl_files[i] for i in nrc_ixes[nrc_val_num:]]
    trn_nrc_img_files = [nrc_img_files[i] for i in nrc_ixes[nrc_val_num:]]

    print('trn_nrc_img, trn_nrc_lbl', len(trn_nrc_img_files), len(trn_nrc_lbl_files))
    print('val_nrc_img, val_nrc_lbl', len(val_nrc_img_files), len(val_nrc_lbl_files))

    bkg_ixes = np.random.permutation(len(bkg_lbl_files))
    trn_non_bkg_num = len(trn_nrc_lbl_files) + len(trn_rc_lbl_files)
    val_non_bkg_num = len(val_nrc_lbl_files) + len(val_rc_lbl_files)
    trn_bkg_lbl_files =[bkg_lbl_files[i] for i in bkg_ixes[:trn_non_bkg_num ]]
    val_bkg_lbl_files = [bkg_lbl_files[i] for i in bkg_ixes[ trn_non_bkg_num: val_non_bkg_num + trn_non_bkg_num]]
    trn_bkg_img_files = [bkg_img_files[i] for i in bkg_ixes[: trn_non_bkg_num]]
    val_bkg_img_files = [bkg_img_files[i] for i in bkg_ixes[trn_non_bkg_num: val_non_bkg_num + trn_non_bkg_num]]
    print('trn_bkg_lbl_files', len(trn_bkg_lbl_files), len(trn_bkg_img_files))
    print('val_bkg_img_files', len(val_bkg_lbl_files), len(val_bkg_img_files))

    nrc_bkg_img_files = val_bkg_img_files + val_nrc_img_files
    val_nrcbkg_img_dir = args.images_save_dir[:-1] + '_val_nrcbkg_img'
    if not os.path.exists(val_nrcbkg_img_dir):
        os.mkdir(val_nrcbkg_img_dir)
    val_nrcbkg_lbl_dir = args.annos_save_dir[:-1] + '_val_nrcbkg_lbl_with_rcid'
    if not os.path.exists(val_nrcbkg_lbl_dir):
        os.mkdir(val_nrcbkg_lbl_dir)
    lbl_bkg_path = args.annos_save_dir[:-1] + '_bkg'
    for f in nrc_bkg_img_files:
        img_name = os.path.basename(f)
        shutil.copy(f, os.path.join(val_nrcbkg_img_dir, img_name))
        lbl_name = img_name.replace('.jpg', '.txt')
        if '_bkg' in lbl_name:
            shutil.copy(os.path.join(lbl_bkg_path, lbl_name), os.path.join(val_nrcbkg_lbl_dir, lbl_name))
        else:
            shutil.copy(os.path.join(lbl_model_path, lbl_name), os.path.join(val_nrcbkg_lbl_dir, lbl_name))

    trn_lbl_files = trn_bkg_lbl_files + trn_nrc_lbl_files # + trn_rc_lbl_files
    val_lbl_files = val_bkg_lbl_files + val_nrc_lbl_files + val_aug_rc_lbl_files
    trn_img_files = trn_bkg_img_files + trn_nrc_img_files # + trn_rc_img_files
    val_img_files = val_bkg_img_files + val_nrc_img_files + val_aug_rc_img_files

    print('trn_num ', len(trn_lbl_files), len(trn_img_files))
    print('val_num ', len(val_lbl_files), len(val_img_files))

    trn_img_txt = open(os.path.join(txt_save_dir, '{}_train_img{}.txt'.format(data_name, comments)), 'w')
    trn_lbl_txt = open(os.path.join(txt_save_dir, '{}_train_lbl{}.txt'.format(data_name, comments)), 'w')
    val_img_txt = open(os.path.join(txt_save_dir, 'xview_ori_nrcbkg_aug_rc_val_img{}.txt'.format(comments)), 'w')
    val_lbl_txt = open(os.path.join(txt_save_dir, 'xview_ori_nrcbkg_aug_rc_val_lbl{}.txt'.format(comments)), 'w')

    trn_lbl_dir = args.data_list_save_dir + comments[1:] + '_trn_lbl'
    val_lbl_dir = args.data_list_save_dir + comments[1:] + '_val_ori_nrcbkg_aug_rc_lbl'
    if os.path.exists(trn_lbl_dir):
        shutil.rmtree(trn_lbl_dir)
        os.mkdir(trn_lbl_dir)
    else:
        os.mkdir(trn_lbl_dir)
    if os.path.exists(val_lbl_dir):
        shutil.rmtree(val_lbl_dir)
        os.mkdir(val_lbl_dir)
    else:
        os.mkdir(val_lbl_dir)

    for i in range(len(trn_lbl_files)):
        trn_lbl_txt.write("%s\n" % trn_lbl_files[i])
        # print('trn_lbl_files', trn_lbl_files[i])
        lbl_name = os.path.basename(trn_lbl_files[i])
        trn_img_txt.write("%s\n" % trn_img_files[i])
        shutil.copy(trn_lbl_files[i], os.path.join(trn_lbl_dir, lbl_name))
    trn_img_txt.close()
    trn_lbl_txt.close()

    for j in range(len(val_lbl_files)):
        # print('val_lbl_files ', j, val_lbl_files[j])
        val_lbl_txt.write("%s\n" % val_lbl_files[j])
        lbl_name = os.path.basename(val_lbl_files[j])
        # print('val_img_files ', j, val_img_files[j])
        val_img_txt.write("%s\n" % val_img_files[j])
        shutil.copy(val_lbl_files[j], os.path.join(val_lbl_dir, lbl_name))
        # exit(0)
    val_img_txt.close()
    val_lbl_txt.close()

    shutil.copyfile(os.path.join(txt_save_dir, '{}_train_img{}.txt'.format(data_name, comments)),
                    os.path.join(data_save_dir, '{}_train_img{}.txt'.format(data_name, comments)))
    shutil.copyfile(os.path.join(txt_save_dir, '{}_train_lbl{}.txt'.format(data_name, comments)),
                    os.path.join(data_save_dir, '{}_train_lbl{}.txt'.format(data_name, comments)))
    shutil.copyfile(os.path.join(txt_save_dir, 'xview_ori_nrcbkg_aug_rc_val_img{}.txt'.format(comments)),
                    os.path.join(data_save_dir, 'xview_ori_nrcbkg_aug_rc_val_img{}.txt'.format(comments)))
    shutil.copyfile(os.path.join(txt_save_dir, 'xview_ori_nrcbkg_aug_rc_val_lbl{}.txt'.format(comments)),
                    os.path.join(data_save_dir, 'xview_ori_nrcbkg_aug_rc_val_lbl{}.txt'.format(comments)))

    data_txt = open(os.path.join(data_save_dir, '{}_aug_rc{}.data'.format(data_name, comments)), 'w')
    data_txt.write(
        'xview_train={}\n'.format(os.path.join(data_save_dir, '{}_train_img{}.txt'.format(data_name, comments))))
    data_txt.write(
        'xview_train_label={}\n'.format(os.path.join(data_save_dir, '{}_train_lbl{}.txt'.format(data_name, comments))))

    data_txt.write(
    'rc_train={}\n'.format(os.path.join(data_save_dir, 'only_rc_train_img{}.txt'.format(comments))))
    data_txt.write(
        'rc_train_label={}\n'.format(os.path.join(data_save_dir, 'only_rc_train_lbl{}.txt'.format(comments))))

    data_txt.write(
        'valid={}\n'.format(os.path.join(data_save_dir, 'xview_ori_nrcbkg_aug_rc_val_img{}.txt'.format(comments))))
    data_txt.write(
        'valid_label={}\n'.format(os.path.join(data_save_dir, 'xview_ori_nrcbkg_aug_rc_val_lbl{}.txt'.format(comments))))

    xview_nrcbkg_img_txt = pd.read_csv(open(os.path.join(data_save_dir, '{}_train_img{}.txt'.format(data_name, comments))), header=None).to_numpy()
    xview_rc_img_txt = pd.read_csv(open(os.path.join(data_save_dir, 'only_rc_train_img{}.txt'.format(comments))), header=None).to_numpy()
    xview_trn_num = xview_nrcbkg_img_txt.shape[0] + xview_rc_img_txt.shape[0]
    data_txt.write('syn_0_xview_number={}\n'.format(xview_trn_num))
    data_txt.write('classes=%s\n' % str(args.class_num))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval=color\n')
    data_txt.close()

def label_m_val_model_with_other_label(rare_class, model_id=1, other_label=0):
    hard_easy_aug_dir = args.annos_save_dir[:-1] + '_rc_val_new_ori_multi_rcid_aug_easy_hard'
    if not os.path.exists(hard_easy_aug_dir):
        os.mkdir(hard_easy_aug_dir)
    des_easy_dir = os.path.join(hard_easy_aug_dir, 'val_aug_m{}_rc{}_easy'.format(model_id, rare_class))
    if not os.path.exists(des_easy_dir):
        os.mkdir(des_easy_dir)
    des_hard_dir = os.path.join(hard_easy_aug_dir, 'val_aug_m{}_rc{}_hard'.format(model_id, rare_class))
    if not os.path.exists(des_hard_dir):
        os.mkdir(des_hard_dir)
    val_labeled_dir = args.annos_save_dir[:-1] + '_rc_val_new_ori_multi_rcid_aug'
    m_model_files = glob.glob(os.path.join(val_labeled_dir, '*.txt'))

    for f in m_model_files:
        lbl_name = os.path.basename(f)
        print('lbl_name', lbl_name)
        # for easy label
        df_easy_txt = pd.read_csv(f, header=None, sep=' ')
        df_easy_txt.loc[df_easy_txt.loc[:, 5] != rare_class, 5] = other_label
        df_easy_txt.to_csv(os.path.join(des_easy_dir, lbl_name), sep=' ', header=False, index=False)
        # for hard label
        df_hard_txt = pd.read_csv(f, header=None, sep=' ')
        length = df_hard_txt.shape[0]
        for t in range(length):
            if df_hard_txt.loc[t, 5] != rare_class:
                df_hard_txt = df_hard_txt.drop(t) # drop index
        df_hard_txt.to_csv(os.path.join(des_hard_dir, lbl_name), sep=' ', header=False, index=False)

    nrcbkg_lbl_path = args.annos_save_dir[:-1] + '_val_nrcbkg_lbl_with_rcid'
    nrcbkg_lbl_files = glob.glob(os.path.join(nrcbkg_lbl_path, '*.txt'))
    for f in nrcbkg_lbl_files:
        if pps.is_non_zero_file(f):
            df_nrcbkg = pd.read_csv(f, header=None, sep=' ')
            df_nrcbkg.loc[:, 5] = other_label
            df_nrcbkg.to_csv(f, sep=' ', header=False, index=False)


def create_val_aug_rc_hard_easy_txt_list_data(model_id, rare_id, pxwhrs='px23whr3_seed17', eh_types=['easy']):
    '''
    create hard easy validation dataset of model* rc*
    '''
    hard_easy_aug_dir = args.annos_save_dir[:-1] + '_rc_val_new_ori_multi_rcid_aug_easy_hard'
    if not os.path.exists(hard_easy_aug_dir):
        os.mkdir(hard_easy_aug_dir)

    # val_labeled_m_rc_hard = os.path.join(hard_easy_aug_dir, '{}_val_lbl_m{}_rc{}_hard'.format(pxwhrs, model_id, rare_id))
    # if not os.path.exists(val_labeled_m_rc_hard):
    #     os.mkdir(val_labeled_m_rc_hard)

    nrcbkg_lbl_path = args.annos_save_dir[:-1] + '_val_nrcbkg_lbl_with_rcid'
    nrcbkg_lbl_files = glob.glob(os.path.join(nrcbkg_lbl_path, '*.txt'))
    nrcbkg_img_path = args.images_save_dir[:-1] +'_val_nrcbkg_img'
    base_dir = os.path.join(args.data_save_dir, pxwhrs, 'RC')
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    for eht in eh_types:
        test_lbl_txt = open(os.path.join(base_dir, 'xview_ori_nrcbkg_aug_rc_test_lbl_{}_m{}_rc{}_{}.txt'.format(pxwhrs, model_id, rare_id, eht)), 'w')
        test_img_txt = open(os.path.join(base_dir, 'xview_ori_nrcbkg_aug_rc_test_img_{}_m{}_rc{}_{}.txt'.format(pxwhrs, model_id, rare_id, eht)), 'w')
        aug_rc_lbl_path = os.path.join(hard_easy_aug_dir, 'val_aug_m{}_rc{}_{}'.format(model_id, rare_id, eht))
        aug_rc_img_path = args.images_save_dir[:-1] + '_rc_val_new_ori_multi_aug'
        aug_rc_lbls = np.sort(glob.glob(os.path.join(aug_rc_lbl_path, '*.txt')))
        for f in aug_rc_lbls:
            test_lbl_txt.write('%s\n' % f)
            name = os.path.basename(f).replace('.txt', '.jpg')
            test_img_txt.write('%s\n' % os.path.join(aug_rc_img_path, name))
        for f in nrcbkg_lbl_files:
            test_lbl_txt.write('%s\n' % f)
            name = os.path.basename(f).replace('.txt', '.jpg')
            test_img_txt.write('%s\n' % os.path.join(nrcbkg_img_path, name))
        test_img_txt.close()
        test_lbl_txt.close()

        data_txt = open(os.path.join(base_dir, 'xview_ori_nrcbkg_aug_rc_test_{}_m{}_rc{}_{}.data'.format(pxwhrs, model_id, rare_id, eht)), 'w')
        data_txt.write('classes=%s\n' % str(args.class_num))
        data_txt.write('test=./data_xview/{}_cls/{}/xview_ori_nrcbkg_aug_rc_test_img_{}_m{}_rc{}_{}.txt\n'.format(args.class_num, pxwhrs,  pxwhrs, model_id, rare_id, eht))
        data_txt.write('test_label=./data_xview/{}_cls/{}/xview_ori_nrcbkg_aug_rc_test_lbl_{}_m{}_rc{}_{}.txt\n'.format(args.class_num, pxwhrs,  pxwhrs, model_id, rare_id, eht))
        data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(args.class_num))
        data_txt.close()


def create_val_aug_rc_nrcbkg_easy_txt_data(pxwhrs='px23whr3_seed17'):
    '''
    create easy validation dataset of all augrc nrcbkg
    all rare classes are classified as one category
    '''
    easy_aug_lbl_dir = args.annos_save_dir[:-1] + '_rc_val_new_ori_multi_rcid_aug'

    nrcbkg_lbl_path = args.annos_save_dir[:-1] + '_val_nrcbkg_lbl_with_rcid'
    nrcbkg_lbl_files = glob.glob(os.path.join(nrcbkg_lbl_path, '*.txt'))
    nrcbkg_img_path = args.images_save_dir[:-1] +'_val_nrcbkg_img'
    base_dir = os.path.join(args.data_save_dir, pxwhrs, 'RC')
    eht = 'easy'
    test_lbl_txt = open(os.path.join(base_dir, 'xview_ori_nrcbkg_aug_rc_test_lbl_{}_all_{}.txt'.format(pxwhrs, eht)), 'w')
    test_img_txt = open(os.path.join(base_dir, 'xview_ori_nrcbkg_aug_rc_test_img_{}_all_{}.txt'.format(pxwhrs, eht)), 'w')
    aug_rc_img_path = args.images_save_dir[:-1] + '_rc_val_new_ori_multi_aug'
    aug_rc_lbls = np.sort(glob.glob(os.path.join(easy_aug_lbl_dir, '*.txt')))
    for f in aug_rc_lbls:
        test_lbl_txt.write('%s\n' % f)
        name = os.path.basename(f).replace('.txt', '.jpg')
        test_img_txt.write('%s\n' % os.path.join(aug_rc_img_path, name))
    for f in nrcbkg_lbl_files:
        test_lbl_txt.write('%s\n' % f)
        name = os.path.basename(f).replace('.txt', '.jpg')
        test_img_txt.write('%s\n' % os.path.join(nrcbkg_img_path, name))
    test_img_txt.close()
    test_lbl_txt.close()

    data_txt = open(os.path.join(base_dir, 'xview_ori_nrcbkg_aug_rc_test_{}_all_{}.data'.format(pxwhrs,  eht)), 'w')
    data_txt.write('classes=%s\n' % str(args.class_num))
    data_txt.write('test=./data_xview/{}_cls/{}/xview_ori_nrcbkg_aug_rc_test_img_{}_all_{}.txt\n'.format(args.class_num, pxwhrs,  pxwhrs, eht))
    data_txt.write('test_label=./data_xview/{}_cls/{}/xview_ori_nrcbkg_aug_rc_test_lbl_{}_all_{}.txt\n'.format(args.class_num, pxwhrs,  pxwhrs, eht))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(args.class_num))
    data_txt.close()


def create_aug_only_rc_testset_txt_list(pxwhrs='px23whr3_seed17'):
    lbl_dir = args.annos_save_dir[:-1] + '_rc_val_new_ori_multi_rcid_aug'
    img_dir = args.images_save_dir[:-1] + '_rc_val_new_ori_multi_aug'
    img_files = glob.glob(os.path.join(img_dir, '*.jpg'))
    img_files.sort()
    base_dir = os.path.join(args.data_save_dir, pxwhrs, 'RC')
    test_lbl_txt = open(os.path.join(base_dir, 'aug_only_rc_test_lbl_{}.txt'.format(pxwhrs)), 'w')
    test_img_txt = open(os.path.join(base_dir, 'aug_only_rc_test_img_{}.txt'.format(pxwhrs)), 'w')

    for f in img_files:
        test_img_txt.write('%s\n' % f)
        test_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(f).replace('.jpg', '.txt')))
    test_img_txt.close()
    test_lbl_txt.close()


def create_train_data_aug_rc_by_rcid(data_name, mid, rcid, comments):
    data_save_dir = os.path.join(args.data_save_dir, comments[1:])
    data_txt = open(os.path.join(data_save_dir, '{}_aug_rc{}{}.data'.format(data_name, rcid, comments)), 'w')
    data_txt.write(
        'xview_train={}\n'.format(os.path.join(data_save_dir, '{}_train_img{}.txt'.format(data_name, comments))))
    data_txt.write(
        'xview_train_label={}\n'.format(os.path.join(data_save_dir, '{}_train_lbl{}.txt'.format(data_name, comments))))
    if rcid:
        data_txt.write(
            'rc_train={}\n'.format(os.path.join(data_save_dir, 'RC', 'only_rc{}_train_img{}.txt'.format(rcid, comments))))
        data_txt.write(
            'rc_train_label={}\n'.format(os.path.join(data_save_dir, 'RC', 'only_rc{}_train_lbl{}.txt'.format(rcid, comments))))
    else:
        data_txt.write(
            'rc_train={}\n'.format(os.path.join(data_save_dir, 'only_rc_train_img{}.txt'.format(comments))))
        data_txt.write(
            'rc_train_label={}\n'.format(os.path.join(data_save_dir, 'only_rc_train_lbl{}.txt'.format(comments))))
    data_txt.write(
        'valid={}\n'.format(os.path.join(data_save_dir, 'RC', '{}_aug_rc_test_img{}_m{}_rc{}_easy.txt'.format(data_name, comments, mid, rcid))))
    data_txt.write(
        'valid_label={}\n'.format(os.path.join(data_save_dir, 'RC', '{}_aug_rc_test_lbl{}_m{}_rc{}_easy.txt'.format(data_name, comments, mid, rcid))))

    xview_nrcbkg_img_txt = pd.read_csv(open(os.path.join(data_save_dir, '{}_train_img{}.txt'.format(data_name, comments))), header=None).to_numpy()
    if rcid:
        xview_rc_img_txt = pd.read_csv(open(os.path.join(data_save_dir, 'RC', 'only_rc{}_train_img{}.txt'.format(rcid, comments))), header=None).to_numpy()
    else:
        xview_rc_img_txt = pd.read_csv(open(os.path.join(data_save_dir, 'only_rc_train_img{}.txt'.format(comments))), header=None).to_numpy()
    xview_trn_num = xview_nrcbkg_img_txt.shape[0] + xview_rc_img_txt.shape[0]
    data_txt.write('syn_0_xview_number={}\n'.format(xview_trn_num))
    data_txt.write('classes=%s\n' % str(args.class_num))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval=color\n')
    data_txt.close()


def get_args(px_thres=None, whr_thres=None): #
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
    parser.add_argument("--val_percent", type=float, default=0.2,
                        help="0.24 0.2 Percent to split into validation (ie .25 = val set is 25% total)")
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

    if not os.path.exists(args.data_save_dir):
        os.makedirs(args.data_save_dir)
    if not os.path.exists(args.data_list_save_dir):
        os.makedirs(args.data_list_save_dir)
    return args


if __name__ == '__main__':
    args = get_args(px_thres=23, whr_thres=3)

    '''
    create multi chips (for one specified tif) and label txt and get all images json, convert from *.geojson to *.json
    w:3475
    h:3197
    '''
    # tif_name = '2315.tif'
    # tif_list = ['86.tif', '88.tif', '311.tif', '546.tif', '1052.tif', '1076.tif', '1114.tif', '2160.tif']
    # for tif_name in tif_list:
    #     get_multi_chips_and_txt_geojson_2_json_of_tif_name(tif_name)

    '''
    manually select desired patches 
    get txt labels for these patches
    group all cropped patches together
    manually label the with rcid !!!!!
    '''
    # tif_list = ['86.tif', '88.tif', '311.tif', '546.tif', '1052.tif', '1076.tif', '1114.tif', '2160.tif', '2315.tif']
    # tif_list = ['1114.tif']
    # for tif_name in tif_list:
    #     get_img_txt_for_multi_chips(tif_name)

    '''
    draw bbox on multi crops of rc with indices
    '''
    # tif_list = ['86.tif', '88.tif', '311.tif', '546.tif', '1052.tif', '1076.tif', '1114.tif', '2160.tif', '2315.tif']
    # # tif_list = ['1114.tif']
    # for tif_name in tif_list:
    #     draw_bbox_with_indices(tif_name)

    '''then scale down the bbox of 1114*.txt'''

    '''
    copy original rc val lbl into the new val folder
    val ori img, val ori lbl
    val multi img, val multi lbl
    original val lbl, img  + multi-crops of val lbl, img
    scale down 1114 bbox first!!!!!!
    drop bbox edge<px_thres or w/h>=whr_thres or h/w>=whr_thres
    '''
    # combine_ori_multi_img_lbl()

    '''
    modify nrcbkg labels 
    change the last column --> 0
    '''
    # annos_dir = args.annos_save_dir[:-1] + '_val_nrcbkg_lbl_with_rcid/'
    # nrc_files = glob.glob(os.path.join(annos_dir, '*.txt'))
    # for f in nrc_files:
    #     if not pps.is_non_zero_file(f):
    #         continue
    #     df_nrc = pd.read_csv(f, header=None, sep=' ')
    #     df_nrc.loc[:, 5] = 0
    #     df_nrc.to_csv(f, header=False, index=False, sep=' ')

    '''
    create data for zero-shot learning
    val: easy: !rare_class ---> other label
    val: hard: !rare_class --> drop
    '''
    # model_ids = [4, 1, 5, 5, 5]
    # rare_ids = [1, 2, 3, 4, 5]
    # for ix, rare_class in enumerate(rare_ids):
    #     model_id = model_ids[ix]
    #     label_m_val_model_with_other_label(rare_class, model_id, other_label=0)

    '''
    split train val with augmented val rc data
    nrc + bkg + augmented rc
    # '''
    # seed = 17
    # comments = '_px23whr3_seed{}'.format(seed)
    # split_trn_val_with_aug_rc_nrcbkg_step_by_step(data_name='xview_ori_nrcbkg', comments=comments, seed=seed)

    '''
    creat m*_rc* test*.txt easy hard
    create .data easy hard
    easy: keep other labels
    hard except rc*, drop others
    '''
    model_ids = [4, 1, 5, 5, 5]
    rare_ids = [1, 2, 3, 4, 5]
    eh_types = ['easy'] # , 'hard'
    for ix, rare_id in enumerate(rare_ids):
        model_id = model_ids[ix]
        create_val_aug_rc_hard_easy_txt_list_data(model_id, rare_id, pxwhrs='px23whr3_seed17')

    '''
    create *.data train on xview_rc, nrcbkg, test on  aug rc with rcid easy
    '''
    seed = 17
    comments = '_px23whr3_seed{}'.format(seed)
    data_name='xview_ori_nrcbkg'
    model_ids = [4, 1, 5, 5, 5]
    rare_ids = [1, 2, 3, 4, 5]
    for ix, rare_id in enumerate(rare_ids):
        model_id = model_ids[ix]
        create_train_data_aug_rc_by_rcid(data_name, model_id, rare_id, comments)

    '''
    creat nrcbkg, aug rc test*.txt easy 
    create .data easy
    easy: keep other labels
    '''
    # create_val_aug_rc_nrcbkg_easy_txt_data(pxwhrs='px23whr3_seed17')

    # create_aug_only_rc_testset_txt_list(pxwhrs='px23whr3_seed17')

    '''
    cheke bbox on images
    '''
    # save_path = args.cat_sample_dir + 'image_with_bbox_indices_val/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # images_dir = args.images_save_dir[:-1] + '_rc_val_new_ori_multi/' # 282
    # annos_dir = args.annos_save_dir[:-1] + '_rc_val_new_ori_multi_rcid/'
    # print('images_dir', images_dir)
    # print('annos_dir', annos_dir)
    # img_list = np.sort(glob.glob(os.path.join(images_dir, '*.jpg')))
    # for img in img_list:
    #     lbl_name = os.path.basename(img).replace('.jpg', '.txt')
    #     lbl_file = os.path.join(annos_dir, lbl_name)
    #     gbc.plot_img_with_bbx(img, lbl_file, save_path, label_index=True)
    #     # gbc.plot_img_with_bbx(img, lbl_file, save_path, rare_id=True)


    # model_ids = [4, 1, 5, 5, 5]
    # rare_ids = [1, 2, 3, 4, 5]
    # aug_rc_annos_dir = args.annos_save_dir[:-1] + '_rc_val_new_ori_multi_rcid_aug_easy_hard/val_aug_m{}_rc{}_hard/'
    # images_dir = args.images_save_dir[:-1] + '_rc_val_new_ori_multi_aug/' # 282
    # print('images_dir', images_dir)
    # img_list = np.sort(glob.glob(os.path.join(images_dir, '*.jpg')))
    # for ix, rid in enumerate(rare_ids):
    #     mid = model_ids[ix]
    #     annos_dir = aug_rc_annos_dir.format(mid, rid)
    #     print('annos_dir', annos_dir)
    #     save_path = args.cat_sample_dir + 'image_with_bbox/val_aug_m{}_rc{}_hard/'.format(mid, rid)
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     for img in img_list:
    #         lbl_name = os.path.basename(img).replace('.jpg', '.txt')
    #         lbl_file = os.path.join(annos_dir, lbl_name)
    #         gbc.plot_img_with_bbx(img, lbl_file, save_path, rare_id=True)



    '''
    manually determine which contains multi-types of models, which should be deleted
    remove label files that contains others type of models
    '''
    # args = get_args()
    # images_dir = args.images_save_dir[:-1] + '_of_2315/m4_2315/' # 282
    # annos_dir = args.annos_save_dir[:-1] + '_of_2315/m4_2315/'
    # image_files = glob.glob(os.path.join(images_dir, '*.jpg'))
    # image_names = [os.path.basename(f).split('.jpg')[0] for f in image_files]
    # anno_files = glob.glob(os.path.join(annos_dir, '*.txt'))
    # for af in anno_files:
    #     lbl_name = os.path.basename(af).split('.')[0]
    #     if lbl_name not in image_names:
    #         os.remove(af)

    '''
    clean and backup annotations with some costraints
    '''
    # px_thres = 15
    # whr_thres = 3
    # args = get_args()
    # tif_name = '2315.tif'
    # txt_norm_dir = args.annos_save_dir[:-1] + '_of_{}/'.format(tif_name.split('.')[0]) + 'm4_{}'.format(tif_name.split('.')[0])
    # print('txt_norm_dir ', txt_norm_dir)
    # pwv.clean_backup_xview_plane_with_constraints(args, txt_norm_dir, px_thres, whr_thres)


    '''
    labeled the bbox with m4
    create lbl list *.txt
    cresate *.data
    '''
    # tif_name = '2315.tif'.split('.')[0]
    # create_testset_txt_list_txt_data_of_tif_name(tif_name)

