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
    create multi chips (for one specified tif) and label txt and get all images json, convert from *.geojson to *.json
    w:3475
    h:3197
    '''
    # tif_name = '2315.tif'
    # get_multi_chips_and_txt_geojson_2_json_of_tif_name(tif_name)

    '''
    cheke bbox on images
    '''
    # # whr_thres = 3 # 3.5
    # # # px_thres= 23
    # # px_thres= 15
    # # args = get_args(px_thres, whr_thres)

    # args = get_args()
    # save_path = args.cat_sample_dir + 'image_with_bbox/m4_2315/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # images_dir = args.images_save_dir[:-1] + '_of_2315/m4_2315/' # 282
    # annos_dir = args.annos_save_dir[:-1] + '_of_2315/m4_2315/'
    # print('images_dir', images_dir)
    # print('annos_dir', annos_dir)
    # img_list = np.sort(glob.glob(os.path.join(images_dir, '*.jpg')))
    # for img in img_list:
    #     lbl_name = os.path.basename(img).replace('.jpg', '.txt')
    #     lbl_file = os.path.join(annos_dir, lbl_name)
    #     gbc.plot_img_with_bbx(img, lbl_file, save_path, label_index=False)


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
