"""
Copyright 2018 Defense Innovation Unit Experimental
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# import tensorflow as tf
import glob
import numpy as np
import argparse
import os
import wv_util as wv
from utils_xview import coord_iou, compute_iou
import pandas as pd
from ast import literal_eval
import json
import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm
import shutil
import cv2
import seaborn as sn
import json


"""
  A script that processes xView imagery. 
  Args:
      image_folder: A folder path to the directory storing xView .tif files
        ie ("xView_data/")

      json_filepath: A file path to the GEOJSON ground truth file
        ie ("xView_gt.geojson")

      val_percent (-t): The percentage of input images to use for test set

      suffix (-s): The suffix for output TFRecord files.  Default suffix 't1' will output
        xview_train_t1.record and xview_test_t1.record

      augment (-a): A boolean value of whether or not to use augmentation

  Outputs:
    Writes two files to the current directory containing training and test data in
        TFRecord format ('xview_train_SUFFIX.record' and 'xview_test_SUFFIX.record')
"""

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def convert_norm(size, box):
    '''
    https://blog.csdn.net/xuanlang39/article/details/88642010
    '''
    dh = 1. / (size[0]) # h--0--y
    dw = 1. / (size[1]) # w--1--x

    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0  # (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert(box):
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]

    return x, y, w, h


def shuffle_images_and_boxes_classes(sh_ims, sh_img_names, sh_box, sh_classes_final, sh_boxids, ks):
    """
    Shuffles images, boxes, and classes, while keeping relative matching indices

    Args:
        sh_ims: an array of images
        sh_img_names: a list of image names
        sh_box: an array of bounding box coordinates ([xmin,ymin,xmax,ymax])
        sh_classes_final: an array of classes
        sh_boxids: a list of bbox ids

    Output:
        Shuffle image, boxes, and classes arrays, respectively
        :param sh_boxids:
        :param ks:
    """
    assert len(sh_ims) == len(sh_box)
    assert len(sh_box) == len(sh_classes_final)

    np.random.seed(0)
    perm = np.random.permutation(len(ks))
    out_m = {}
    out_b = {}
    out_c = {}
    out_n = {}
    out_i = {}

    c = 0
    for ind in perm:
        out_m[c] = sh_ims[ks[ind]]
        out_b[c] = sh_box[ks[ind]]
        out_c[c] = sh_classes_final[ks[ind]]
        out_n[c] = sh_img_names[ks[ind]]
        out_i[c] = sh_boxids[ks[ind]]
        c = c + 1
    return out_m, out_n, out_b, out_c, out_i


def create_chips_and_txt_geojson_2_json():
    coords, chips, classes, features_ids = wv.get_labels(args.json_filepath, args.class_num)
    # gs = json.load(open('/media/lab/Yang/data/xView/xView_train.geojson'))
    res = (args.input_size, args.input_size)

    file_names = glob.glob(args.image_folder + "*.tif")
    file_names.sort()

    #fixme
    img_save_dir = args.images_save_dir
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)

    df_img_num_names = pd.DataFrame(columns=['id', 'file_name'])

    txt_norm_dir = args.annos_save_dir
    if not os.path.exists(txt_norm_dir):
        os.makedirs(txt_norm_dir)
    image_names_list = []
    _img_num = 0
    img_num_list = []
    image_info_list = []
    annotation_list = []

    for f_name in tqdm(file_names):
        # Needs to be "X.tif", ie ("5.tif")
        # Be careful!! Depending on OS you may need to change from '/' to '\\'.  Use '/' for UNIX and '\\' for windows
        arr = wv.get_image(f_name)
        name = f_name.split("/")[-1]
        '''
         # drop 1395.tif
        '''
        if name == '1395.tif':
            continue
        ims, img_names, box, classes_final, box_ids = wv.chip_image(arr, coords[chips == name],
                                                                    classes[chips == name],
                                                                    features_ids[chips == name], res,
                                                                    name.split('.')[0], img_save_dir)
        if not img_names:
            continue

        ks = [k for k in ims.keys()]

        for k in ks:
            file_name = img_names[k]
            image_names_list.append(file_name)
            ana_txt_name = file_name.split(".")[0] + ".txt"
            f_txt = open(os.path.join(args.annos_save_dir, ana_txt_name), 'w')
            img = wv.get_image(args.images_save_dir + file_name)
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
                    # "image_name": img_name, #fixme: there aren't 'image_name'
                    "category_id": np.int(classes_final[k][d]),
                    "iscrowd": 0,
                    "area": (bbx[2] - bbx[0] + 1) * (bbx[3] - bbx[1] + 1),
                    "bbox": [bbx[0], bbx[1], bbx[2] - bbx[0], bbx[3] - bbx[1]], # (w1, h1, w, h)
                    "segmentation": [],
                }
                annotation_list.append(annotation_info)

                bbx = [np.int(b) for b in box[k][d]]
                cvt_bbx = convert_norm(res, bbx)
                f_txt.write("%s %s %s %s %s\n" % (np.int(classes_final[k][d]), cvt_bbx[0], cvt_bbx[1], cvt_bbx[2], cvt_bbx[3]))
            img_num_list.append(_img_num)
            _img_num += 1
            f_txt.close()
    df_img_num_names['id'] = img_num_list
    df_img_num_names['file_name'] = image_names_list
    df_img_num_names.to_csv(os.path.join(args.txt_save_dir, 'image_names_{}_{}cls.csv'.format(args.input_size, args.class_num)))

    trn_instance = {'info': 'xView all chips 600 yx185 created', 'license': ['license'], 'images': image_info_list,
                    'annotations': annotation_list, 'categories': wv.get_all_categories(args.class_num)}
    json_file = os.path.join(args.txt_save_dir, 'xViewall_{}_{}cls_xtlytlwh.json'.format(args.input_size, args.class_num)) # topleft
    json.dump(trn_instance, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)


def create_xview_names():
    df_cat = pd.read_csv(args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num), delimiter='\t')
    cat_names = df_cat['category'].to_list()
    f_txt = open(os.path.join(args.data_save_dir, 'xview.names'), 'w')
    for i in range(len(cat_names)):
        f_txt.write("%s\n" % cat_names[i])
    f_txt.close()


def split_trn_val_with_chips():

    all_files = glob.glob(args.images_save_dir + '*.jpg')
    lbl_path = args.annos_save_dir
    num_files = len(all_files)
    trn_num = int(num_files * (1 - args.val_percent))
    np.random.seed(args.seed)
    perm_files = np.random.permutation(all_files)

    trn_img_txt = open(os.path.join(args.txt_save_dir, 'xviewtrain_img.txt'), 'w')
    trn_lbl_txt = open(os.path.join(args.txt_save_dir, 'xviewtrain_lbl.txt'), 'w')
    val_img_txt = open(os.path.join(args.txt_save_dir, 'xviewval_img.txt'), 'w')
    val_lbl_txt = open(os.path.join(args.txt_save_dir, 'xviewval_lbl.txt'), 'w')

    for i in range(trn_num):
        trn_img_txt.write("%s\n" % perm_files[i])
        img_name = perm_files[i].split('/')[-1]
        lbl_name = img_name.replace('.jpg', '.txt')
        trn_lbl_txt.write("%s\n" % (lbl_path + lbl_name))

    trn_img_txt.close()
    trn_lbl_txt.close()

    for i in range(trn_num, num_files):
        val_img_txt.write("%s\n" % perm_files[i])
        img_name = perm_files[i].split('/')[-1]
        lbl_name = img_name.replace('.jpg', '.txt')
        val_lbl = os.path.join(lbl_path, lbl_name)
        val_lbl_txt.write("%s\n" % val_lbl)

    val_img_txt.close()
    val_lbl_txt.close()

    shutil.copyfile(os.path.join(args.txt_save_dir, 'xviewtrain_img.txt'), os.path.join(args.data_save_dir, 'xviewtrain_img.txt'))
    shutil.copyfile(os.path.join(args.txt_save_dir, 'xviewtrain_lbl.txt'), os.path.join(args.data_save_dir, 'xviewtrain_lbl.txt'))
    shutil.copyfile(os.path.join(args.txt_save_dir, 'xviewval_img.txt'), os.path.join(args.data_save_dir, 'xviewval_img.txt'))
    shutil.copyfile(os.path.join(args.txt_save_dir, 'xviewval_lbl.txt'), os.path.join(args.data_save_dir, 'xviewval_lbl.txt'))


def split_trn_val_with_tifs():

    tif_pre_names = [os.path.basename(t).split('.')[0] for t in glob.glob(args.image_folder + '*.tif')]
    tif_num = len(tif_pre_names)
    np.random.seed(args.seed)
    perm_pre_tif_files = np.random.permutation(tif_pre_names)
    trn_tif_num = int(tif_num * (1 - args.val_percent))

    all_files = glob.glob(args.images_save_dir + '*.jpg')
    lbl_path = args.annos_save_dir
    num_files = len(all_files)

    trn_img_txt = open(os.path.join(args.txt_save_dir, 'xviewtrain_img_tifsplit.txt'), 'w')
    trn_lbl_txt = open(os.path.join(args.txt_save_dir, 'xviewtrain_lbl_tifsplit.txt'), 'w')
    val_img_txt = open(os.path.join(args.txt_save_dir, 'xviewval_img_tifsplit.txt'), 'w')
    val_lbl_txt = open(os.path.join(args.txt_save_dir, 'xviewval_lbl_tifsplit.txt'), 'w')

    for i in range(trn_tif_num):
        for j in range(num_files):
            jpg_name = all_files[j].split('/')[-1]
            if jpg_name.startswith(perm_pre_tif_files[i]):
                trn_img_txt.write("%s\n" % all_files[j])
                img_name = all_files[j].split('/')[-1]
                lbl_name = img_name.replace('.jpg', '.txt')
                trn_lbl_txt.write("%s\n" % (lbl_path + lbl_name))

    trn_img_txt.close()
    trn_lbl_txt.close()

    for i in range(trn_tif_num, tif_num):
        for j in range(num_files):
            jpg_name = all_files[j].split('/')[-1]
            if jpg_name.startswith(perm_pre_tif_files[i]):
                val_img_txt.write("%s\n" % all_files[j])
                img_name = all_files[j].split('/')[-1]
                lbl_name = img_name.replace('.jpg', '.txt')
                val_lbl_txt.write("%s\n" % (lbl_path + lbl_name))

    val_img_txt.close()
    val_lbl_txt.close()

    shutil.copyfile(os.path.join(args.txt_save_dir, 'xviewtrain_img_tifsplit.txt'), os.path.join(args.data_save_dir, 'xviewtrain_img_tifsplit.txt'))
    shutil.copyfile(os.path.join(args.txt_save_dir, 'xviewtrain_lbl_tifsplit.txt'), os.path.join(args.data_save_dir, 'xviewtrain_lbl_tifsplit.txt'))
    shutil.copyfile(os.path.join(args.txt_save_dir, 'xviewval_img_tifsplit.txt'), os.path.join(args.data_save_dir, 'xviewval_img_tifsplit.txt'))
    shutil.copyfile(os.path.join(args.txt_save_dir, 'xviewval_lbl_tifsplit.txt'), os.path.join(args.data_save_dir, 'xviewval_lbl_tifsplit.txt'))


def create_json_for_train_or_val_according_to_all_json(typestr='val'):
    json_all_file = os.path.join(args.txt_save_dir, 'xViewall_{}_{}cls_xtlytlwh.json'.format(args.input_size, args.class_num)) # topleft
    all_json = json.load(open(json_all_file))
    all_imgs_info = all_json['images']
    all_annos_info = all_json['annotations']
    all_cats_info = all_json['categories']

    all_img_id_maps = json.load(open(args.txt_save_dir + 'all_image_ids_names_dict_{}cls.json'.format(args.class_num)))
    all_imgs = [i for i in all_img_id_maps.values()]
    all_img_ids = [int(i) for i in all_img_id_maps.keys()]

    val_img_files = pd.read_csv(os.path.join(args.data_save_dir, 'xview{}_img.txt'.format(typestr)), header=None)
    val_img_names = [f.split('/')[-1] for f in val_img_files[0].tolist()]
    val_img_ids = [all_img_ids[all_imgs.index(v)] for v in val_img_names]

    image_info_list = []
    anno_info_list = []

    for ix in val_img_ids:
        img_name = all_imgs[ix]
        # for i in all_imgs_info:
        #     if all_imgs_info[i]['id'] == ix:
        #fixme: the index of all_imgs_info is the same as the image_info['id']
        image_info_list.append(all_imgs_info[ix])
        for ai in all_annos_info:
            if ai['image_id'] == ix:
                anno_info_list.append(ai)

    trn_instance = {'info': 'xView val chips 600 yx185 created', 'license': ['license'], 'images': image_info_list,
                    'annotations': anno_info_list, 'categories': all_cats_info}
    json_file = os.path.join(args.txt_save_dir, 'xView{}_{}_{}cls_xtlytlwh.json'.format(typestr, args.input_size, args.class_num)) # topleft
    json.dump(trn_instance, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)


def get_train_or_val_imgs_by_cat_id(cat_ids, typestr='val'):
    cat_img_ids_maps = json.load(open(os.path.join(args.txt_save_dir, 'all_cat_img_ids_dict_{}cls.json'.format(args.class_num))))
    img_ids_names_maps = json.load(open(os.path.join(args.txt_save_dir, 'all_image_ids_names_dict_{}cls.json'.format(args.class_num))))
    imgids = [k for k in img_ids_names_maps.keys()]
    imgnames = [v for v in img_ids_names_maps.values()]

    val_files = pd.read_csv(os.path.join(args.data_save_dir, 'xview{}_img.txt'.format(typestr)), header=None)
    val_img_names = [f.split('/')[-1] for f in val_files[0]]
    val_img_ids = [imgids[imgnames.index(n)] for n in val_img_names]
    cat_val_imgs = {}

    all_cats_val_imgs = []
    for rc in cat_ids:
        cat_img_ids = cat_img_ids_maps[rc]
        cat_img_names = [val_img_names[val_img_ids.index(str(i))] for i in cat_img_ids if str(i) in val_img_ids]
        all_cats_val_imgs.extend(cat_img_names)
        cat_val_imgs[rc] = cat_img_names
    cat_val_imgs_files = os.path.join(args.txt_save_dir, 'cat_ids_2_{}_img_names.json'.format(typestr))
    json.dump(cat_val_imgs, open(cat_val_imgs_files, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)


def create_xview_data():
    data_txt = open(os.path.join(args.data_save_dir, 'xview.data'), 'w')
    data_txt.write('classes=%s\n' % str(args.class_num))
    data_txt.write('train=./data_xview/{}_cls/xviewtrain_img.txt\n'.format(args.class_num))
    data_txt.write('train_label=./data_xview/{}_cls/xviewtrain_lbl.txt\n'.format(args.class_num))
    data_txt.write('valid=./data_xview/{}_cls/xviewval_img.txt\n'.format(args.class_num))
    data_txt.write('valid_label=./data_xview/{}_cls/xviewval_lbl.txt\n'.format(args.class_num))
    data_txt.write('valid_rare=./data_xview/{}_cls/xviewval_rare_img.txt\n'.format(args.class_num))
    data_txt.write('valid_rare_label=./data_xview/{}_cls/xviewval_rare_lbl.txt\n'.format(args.class_num))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval=xview')
    data_txt.close()


def plot_val_image_with_gt_bbx_by_image_name_from_patches(image_name):
    img_ids_names_file = args.txt_save_dir + 'all_image_ids_names_dict_{}cls.json'.format(args.class_num)
    img_ids_names_map = json.load(open(img_ids_names_file))
    img_ids = [int(k) for k in img_ids_names_map.keys()] ## important
    img_names = [v for v in img_ids_names_map.values()]

    df_cat_color = pd.read_csv(args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num, args.class_num), delimiter='\t')
    cat_ids = df_cat_color['category_id'].tolist()
    cat_colors = df_cat_color['color'].tolist()

    df_val_img = pd.read_csv(args.data_save_dir + 'xviewval_img.txt'.format(args.class_num), header=None)
    df_val_gt = pd.read_csv(args.data_save_dir + 'xviewval_lbl.txt'.format(args.class_num), header=None)
    val_img_list = df_val_img[0].tolist()
    img_size = 608

    val_img_names = [f.split('/')[-1] for f in val_img_list]
    img_index = val_img_names.index(image_name)
    gt = pd.read_csv(df_val_gt[0].iloc[img_index], header=None, delimiter=' ')
    gt.iloc[:, 1:] = gt.iloc[:, 1:]*img_size
    gt.iloc[:, 1] = gt.iloc[:, 1] - gt.iloc[:, 3]/2
    gt.iloc[:, 2] = gt.iloc[:, 2] - gt.iloc[:, 4]/2

    img_bbx_fig_dir = args.cat_sample_dir + 'img_name_2_gt_bbx_figures/'
    if not os.path.exists(img_bbx_fig_dir):
        os.makedirs(img_bbx_fig_dir)

    img = cv2.imread(val_img_list[img_index])
    for ix in range(len(gt)):
        cat_id = gt.iloc[ix, 0]
        color = literal_eval(cat_colors[cat_ids.index(cat_id)])
        gt_bbx = gt.iloc[ix, 1:].to_numpy()
        gt_bbx = gt_bbx.astype(np.int64)
        img = cv2.rectangle(img, (gt_bbx[0], gt_bbx[1]), (gt_bbx[0] + gt_bbx[2], gt_bbx[1] + gt_bbx[3]), color, 2)
        # cv2.putText(img, text=str(cat_id), org=(gt_bbx[0] + 10, gt_bbx[1] + 10),
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 255))
    cv2.imwrite(img_bbx_fig_dir + image_name, img)


def plot_img_with_prd_bbx_by_image_name(image_name, cat_ids, score_thres=0.3, whr_thres=3):
    img_id = get_img_id_by_image_name(image_name)

    results = json.load(open(args.results_dir + 'results.json'))
    results_img_id_cat_id = [rs for rs in results if rs['category_id'] in cat_ids and rs['image_id'] == img_id]
    save_dir = args.cat_sample_dir + 'img_name_2_prd_bbx_figures/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    img = cv2.imread(args.images_save_dir + image_name)
    for rs in results_img_id_cat_id:
        cat_id = rs['category_id']
        w, h = rs['bbox'][2:]
        whr = np.maximum(w/(h+1e-16), h/(w+1e-16))
        rs['bbox'][2] = rs['bbox'][2] + rs['bbox'][0]
        rs['bbox'][3] = rs['bbox'][3] + rs['bbox'][1]
        # print('whr', whr)
        if whr <= whr_thres and rs['score'] >= score_thres:
            bbx = [int(x) for x in rs['bbox']]
            cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[2], bbx[3]), (0, 255, 255), 2)
            cv2.putText(img, text='{} {:.3f}'.format(cat_id, rs['score']), org=(bbx[0] - 5, bbx[1] - 5), # [pr_bx[0], pr[-1]]
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 0))
    cv2.imwrite(save_dir + 'cat{}_'.format(cat_ids) + image_name, img)


def get_img_id_by_image_name(image_name):
    img_ids_names_file = args.txt_save_dir + 'all_image_ids_names_dict_{}cls.json'.format(args.class_num)
    img_ids_names_map = json.load(open(img_ids_names_file))
    img_ids = [int(k) for k in img_ids_names_map.keys()] ## important
    img_names = [v for v in img_ids_names_map.values()]
    return img_ids[img_names.index(image_name)]

#fixme
# def plot_rare_results_by_rare_cat_ids(rare_cat_ids, score_thres=0.001):
#     '''
#     all: TP + FP + FN
#     '''
#     img_ids_names_file = '/media/lab/Yang/data/xView_YOLO/labels/{}/all_image_ids_names_dict_{}cls.json'.format(args.input_size, args.class_num)
#     img_ids_names_map = json.load(open(img_ids_names_file))
#     img_ids = [int(k) for k in img_ids_names_map.keys()] ## important
#     img_names = [v for v in img_ids_names_map.values()]
#
#     df_rare_val_img = pd.read_csv('../data_xview/{}_cls/xviewval_rare_img.txt'.format(args.class_num), header=None)
#     df_rare_val_gt = pd.read_csv('../data_xview/{}_cls/xviewval_rare_lbl.txt'.format(args.class_num), header=None)
#     rare_val_img_list = df_rare_val_img[0].tolist()
#     rare_val_id_list = df_rare_val_gt[0].tolist()
#     rare_val_img_names = [f.split('/')[-1] for f in rare_val_img_list]
#     rare_val_img_ids = [img_ids[img_names.index(x)] for x in rare_val_img_names]
#
#     rare_result_json_file = '../result_output/{}_cls/results_rare.json'.format(args.class_num)
#     rare_result_allcat_list = json.load(open(rare_result_json_file))
#     rare_result_list = []
#     for ri in rare_result_allcat_list:
#         if ri['category_id'] in rare_cat_ids and ri['score'] >= score_thres:
#             rare_result_list.append(ri)
#     del rare_result_allcat_list
#
#     prd_color = (255, 255, 0)
#     gt_color = (0, 255, 255) # yellow
#     save_dir = '../result_output/rare_result_figures/'
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
#
#     for ix in range(len(rare_val_img_list)):
#         img = cv2.imread(rare_val_img_list[ix])
#         img_size = img.shape[0]
#         gt_rare_cat = pd.read_csv(rare_val_id_list[ix], delimiter=' ').to_numpy()
#         gt_rare_list = []
#         # for cat_id in rare_cat_ids:
#         #     for gx in range(gt_rare_cat.shape[0]):
#         #         if gt_rare_cat[gx, 0] == cat_id:
#         #             gt_rare_cat[gx, 1:] = gt_rare_cat[gx, 1:] * img_size
#         #             gt_rare_cat[gx, 1] = gt_rare_cat[gx, 1] - gt_rare_cat[gx, 3]/2
#         #             gt_rare_cat[gx, 2] = gt_rare_cat[gx, 2] - gt_rare_cat[gx, 4]/2
#         #             gt_bbx = gt_rare_cat[gx, 1:]
#         #             gt_bbx = gt_bbx.astype(np.int64)
#         #             img = cv2.rectangle(img, (gt_bbx[0], gt_bbx[1]), (gt_bbx[0] + gt_bbx[2], gt_bbx[1] + gt_bbx[3]), gt_color, 2)
#         #             cv2.putText(img, text=str(cat_id), org=(gt_bbx[0] - 10, gt_bbx[1] - 10),
#         #                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#         #                         fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=gt_color)
#         #             gt_rare_list.append(gx)
#         #     img_prd_lbl_list = []
#         #     for jx in rare_result_list:
#         #         if jx['category_id'] == cat_id and jx['image_id'] == rare_val_img_ids[ix]:
#         #             img_prd_lbl_list.append(jx) # rare_result_list.pop()
#         #             bbx = jx['bbox'] # xtopleft, ytopleft, w, h
#         #             bbx = [int(x) for x in bbx]
#         #             img = cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[0] + bbx[2], bbx[1] + bbx[3]), prd_color, 2)
#         #             cv2.putText(img, text=str(jx['category_id']), org=(bbx[0] + 10, bbx[1] + 20),
#         #                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#         #                         fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=prd_color)
#         #     if gt_rare_list or img_prd_lbl_list:
#         #         cv2.imwrite(save_dir + 'cat{}_'.format(cat_id) + rare_val_img_names[ix], img)
#         for gx in range(gt_rare_cat.shape[0]):
#             cat_id = int(gt_rare_cat[gx, 0])
#             if cat_id in rare_cat_ids:
#                 gt_rare_cat[gx, 1:] = gt_rare_cat[gx, 1:] * img_size
#                 gt_rare_cat[gx, 1] = gt_rare_cat[gx, 1] - gt_rare_cat[gx, 3]/2
#                 gt_rare_cat[gx, 2] = gt_rare_cat[gx, 2] - gt_rare_cat[gx, 4]/2
#                 gt_bbx = gt_rare_cat[gx, 1:]
#                 gt_bbx = gt_bbx.astype(np.int64)
#                 img = cv2.rectangle(img, (gt_bbx[0], gt_bbx[1]), (gt_bbx[0] + gt_bbx[2], gt_bbx[1] + gt_bbx[3]), gt_color, 2)
#                 cv2.putText(img, text=str(cat_id), org=(gt_bbx[0] - 10, gt_bbx[1] - 10),
#                             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                             fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=gt_color)
#                 gt_rare_list.append(gx)
#         img_prd_lbl_list = []
#         for jx in rare_result_list:
#             jx_cat_id = jx['category_id']
#             if jx['image_id'] == rare_val_img_ids[ix] and jx_cat_id in rare_cat_ids:
#                 img_prd_lbl_list.append(jx) # rare_result_list.pop()
#                 bbx = jx['bbox'] # xtopleft, ytopleft, w, h
#                 bbx = [int(x) for x in bbx]
#                 img = cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[0] + bbx[2], bbx[1] + bbx[3]), prd_color, 2)
#                 cv2.putText(img, text=str(jx_cat_id), org=(bbx[0] + 10, bbx[1] + 20),
#                             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                             fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=prd_color)
#
#         if gt_rare_list or img_prd_lbl_list:
#             cv2.imwrite(save_dir + 'rare_cat_' + rare_val_img_names[ix], img)


def cats_split_crop_bbx_from_patches(typestr='val', whr_thres=3, N=200):
    img_files = pd.read_csv(os.path.join(args.data_save_dir, 'xview{}_img.txt'.format(typestr)), header=None)
    lbl_files = pd.read_csv(os.path.join(args.data_save_dir, 'xview{}_lbl.txt'.format(typestr)), header=None)
    lbl_files = lbl_files[0].to_list()
    img_names = [f.split('/')[-1] for f in img_files[0].tolist()]
    # img_ids = [all_img_ids[all_imgs.index(v)] for v in img_names]
    df_cats = pd.read_csv(args.data_save_dir + 'xview.names'.format(args.class_num), header=None)
    cat_names = df_cats.iloc[:, 0]
    for cx, c in enumerate(cat_names):
        cat_dir = args.cat_bbx_patches_dir + 'cat_{}_{}_bbxes/'.format(cx, c)
        if not os.path.exists(os.path.join(cat_dir)):
            os.mkdir(cat_dir)

    for cx, c in enumerate(cat_names):
        cntN = 0
        cat_dir = args.cat_bbx_patches_dir + 'cat_{}_{}_bbxes/'.format(cx, c)
        for ix, name in enumerate(img_names):
            #fixme
            # save first N samples
            cntN += 1
            if cntN > N:
                break
            df_lbl = pd.read_csv(lbl_files[ix], header=None, delimiter=' ')
            df_lbl.iloc[:, 1:] = df_lbl.iloc[:, 1:] * args.input_size
            df_lbl.iloc[:, 1] = df_lbl.iloc[:, 1] - df_lbl.iloc[:, 3]/2
            df_lbl.iloc[:, 2] = df_lbl.iloc[:, 2] - df_lbl.iloc[:, 4]/2
            df_lbl.iloc[:, 3] = df_lbl.iloc[:, 1] + df_lbl.iloc[:, 3]
            df_lbl.iloc[:, 4] = df_lbl.iloc[:, 2] + df_lbl.iloc[:, 4]
            df_lbl = df_lbl.to_numpy().astype(np.int)

            cat_bxs = df_lbl[df_lbl[:, 0] == cx]
            img = cv2.imread(args.images_save_dir + name)
            for bx, cb in enumerate(cat_bxs):
                bbx = cb[1:]
                w = bbx[2] - bbx[0]
                h = bbx[3] - bbx[1]
                whr = np.maximum(w/(h+1e-16), h/(w+1e-16))
                if whr > whr_thres or w < 4 or h < 4:
                    continue
                bbx_img = img[bbx[1]:bbx[3], bbx[0]:bbx[2], :] # h w c
                cv2.imwrite(cat_dir + name.split('.')[0] + '_cat{}_{}.jpg'.format(cx, bx), bbx_img)


def cats_split_crop_bbx_from_origins(whr_thres=3, imgN=200, N=10):
    if args.rare:
        df_category_id = pd.read_csv(args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num), sep="\t")
    else:
        df_category_id = pd.read_csv(args.data_save_dir + 'categories_id_color_{}_new_group.txt'.format(args.class_num), sep="\t")
    category_ids = df_category_id['category_id'].tolist()
    category_labels = df_category_id['category_label'].tolist()
    category_names = df_category_id['category'].tolist()
    cat_names = list(set(category_names))

    data = json.load(open(args.json_filepath))
    feas = data['features']
    img_names = []
    fea_indices = []
    cat_imgs_dict = {}
    cat_wh_dict = {}
    for cx, cn in enumerate(cat_names):
        cat_imgs_dict[cx] = []
        cat_wh_dict[cx] = []
        if not os.path.exists(args.cat_bbx_origins_dir + 'cat_{}_{}_bbxes/'.format(cx, cn)):
            os.mkdir(args.cat_bbx_origins_dir + 'cat_{}_{}_bbxes/'.format(cx, cn))

    for i in range(len(feas)):
        clbl = feas[i]['properties']['type_id']
        if clbl in category_labels:
            img_name = feas[i]['properties']['image_id']
            cid = category_ids[category_labels.index(clbl)]
            if img_name not in cat_imgs_dict[cid]:
                cat_imgs_dict[cid].append(img_name)
            img_names.append(img_name)
            fea_indices.append(i)

    json_file = os.path.join(args.ori_lbl_dir, 'xView_cid_2_imgs_maps.json') # topleft
    json.dump(cat_imgs_dict, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)

    img_names = list(set(img_names))
    img_cid_bbxs_dict = {}
    for n in img_names:
        img_cid_bbxs_dict[n] = []

    for fx in fea_indices:
        clbl = feas[fx]['properties']['type_id']
        cid = category_ids[category_labels.index(clbl)]
        bbx = list(literal_eval(feas[fx]['properties']['bounds_imcoords']))  # (w1, h1, w2, h2)
        img_cid_bbxs_dict[feas[fx]['properties']['image_id']].append([cid] + bbx) # [cid, w1,h1, w2, h2]

    json_file = os.path.join(args.ori_lbl_dir, 'xView_img_2_cid_bbx_maps.json') # topleft
    json.dump(img_cid_bbxs_dict, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)

    del data, feas

    for cx, cn in enumerate(cat_names):
        print('category_id', cx)
        img_list = cat_imgs_dict[cx]
        # fixme show imgN images
        for ix, name in enumerate(img_list):
            cntN = 0
            img = cv2.imread(args.image_folder + name) # 2355.tif
            cbbx_list = img_cid_bbxs_dict[name] # [[18, 2726, 2512, 2740, 2518], [18, 2729, 2494, 2737, 2504]]
            # print(name, cbbx_list)
            for i in range(len(cbbx_list)):
                bbx = cbbx_list[i][1:]
                w = bbx[2] - bbx[0]
                h = bbx[3] - bbx[1]
                whr = np.maximum(w/(h+1e-16), h/(w+1e-16))
                if cbbx_list[i][0] == cx and whr <= whr_thres and w >= 4 and h >= 4:
                    cat_wh_dict[cx].append([w, h])
                    #fixme
                    # each image select N samples
                    if ix < imgN and cntN < N:
                        cat_dir = args.cat_bbx_origins_dir + 'cat_{}_{}_bbxes/'.format(cx, cn)
                        bbx_img = img[bbx[1]:bbx[3], bbx[0]:bbx[2], :] # h w c
                        cv2.imwrite(cat_dir + name.split('.')[0] + '_cat{}_{}.jpg'.format(cx, i), bbx_img)
                    cntN += 1
    json_file = os.path.join(args.ori_lbl_dir, 'xView_cid_2_wh_maps.json') #
    json.dump(cat_wh_dict, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)


def plot_image_with_gt_bbx_by_image_name_from_origin(img_name):
    df_cat_color = pd.read_csv(args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num), delimiter='\t')
    cat_ids = df_cat_color['category_id'].tolist()
    cat_colors = df_cat_color['color'].tolist()

    img_name_cid_maps = json.load(open(args.ori_lbl_dir + 'xView_img_2_cid_bbx_maps.json'))
    img_bbx_list = img_name_cid_maps[img_name]

    img_bbx_fig_dir = args.cat_sample_dir + 'img_name_2_gt_bbx_figures/'
    if not os.path.exists(img_bbx_fig_dir):
        os.makedirs(img_bbx_fig_dir)

    img = cv2.imread(args.image_folder + img_name)
    h, w, _ = img.shape
    for lbl in img_bbx_list:
        print(lbl)
        cat_id = lbl[0]
        cat_color = literal_eval(cat_colors[cat_ids.index(cat_id)])
        lbl[1] = max(0, lbl[1]) # w1
        lbl[2] = max(0, lbl[2]) # h1
        lbl[3] = min(w, lbl[3]) # w2
        lbl[4] = min(h, lbl[4]) # h2

        img = cv2.rectangle(img, (lbl[1], lbl[2]), (lbl[3], lbl[4]), cat_color, 2)
    cv2.imwrite(img_bbx_fig_dir + img_name, img)


def draw_wh_scatter_for_cats(cid_wh_maps, cat_ids):
    df_category_id = pd.read_csv(args.data_save_dir + 'categories_id_color_{}_new_group.txt'.format(args.class_num), sep="\t")
    category_ids = df_category_id['category_id'].tolist()
    categories = df_category_id['category'].tolist()

    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    for cx, cid in enumerate(cat_ids):
        wh_arr = np.array(cid_wh_maps[str(cid)])
        plt.scatter(wh_arr[:, 0], wh_arr[:, 1], label='{}'.format(categories[category_ids.index(cid)]))

    plt.legend(prop=literal_eval(args.font3), loc='upper right')

    xlabel = 'Width'
    ylabel = "Height"

    plt.title('Categories W H Distribution', literal_eval(args.font2))
    plt.ylabel(ylabel, literal_eval(args.font2))
    plt.xlabel(xlabel, literal_eval(args.font2))
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(args.ori_lbl_dir, 'cats_wh_distribution.jpg'))
    plt.show()


def check_prd_gt_iou(image_name, score_thres=0.3, iou_thres=0.5, rare=False):
    img_id = get_img_id_by_image_name(image_name)
    img = cv2.imread(args.images_save_dir + image_name)
    img_size = img.shape[0]
    gt_rare_cat = pd.read_csv(args.annos_save_dir + image_name.replace('.jpg', '.txt'), header=None, delimiter=' ')
    gt_rare_cat = gt_rare_cat.to_numpy()
    gt_rare_cat[:, 1:] = gt_rare_cat[:, 1:] * img_size
    gt_rare_cat[:, 1] = gt_rare_cat[:, 1] - gt_rare_cat[:, 3]/2
    gt_rare_cat[:, 2] = gt_rare_cat[:, 2] - gt_rare_cat[:, 4]/2
    gt_rare_cat[:, 3] = gt_rare_cat[:, 1] + gt_rare_cat[:, 3]
    gt_rare_cat[:, 4] = gt_rare_cat[:, 2] + gt_rare_cat[:, 4]
    if rare:
        prd_lbl_rare = json.load(open(args.results_dir + 'results_rare.json'.format(args.class_num))) # xtlytlwh
    else:
        prd_lbl_rare = json.load(open(args.results_dir + 'results.json'.format(args.class_num))) # xtlytlwh
    for px, p in enumerate(prd_lbl_rare):
        if p['image_id'] == img_id and p['score'] >= score_thres:
            p_bbx = p['bbox'] # xtlytlwh
            p_bbx[2] = p_bbx[0] + p_bbx[2]
            p_bbx[3] = p_bbx[3] + p_bbx[1]
            p_bbx = [int(x) for x in p_bbx]
            p_cat_id = p['category_id']
            g_lbl_part = gt_rare_cat[gt_rare_cat[:, 0] == p_cat_id, :]
            for g in g_lbl_part:
                g_bbx = [int(x) for x in g[1:]]
                iou = coord_iou(p_bbx, g[1:])
                # print('iou', iou)
                if iou >= iou_thres:
                    print(iou)
                    img = cv2.rectangle(img, (p_bbx[0], p_bbx[1]), (p_bbx[2], p_bbx[3]), (0, 0, 255), 2)
                    cv2.putText(img, text='[{}, {:.3f}]'.format(p_cat_id, iou), org=(p_bbx[0] - 10, p_bbx[1] - 10), # [pr_bx[0], pr[-1]]
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 0))
                    img = cv2.rectangle(img, (g_bbx[0], g_bbx[1]), (g_bbx[2], g_bbx[3]), (255, 255, 0), 2) # cyan
    if rare:
        rare_result_iout_check_dir = args.results_dir + 'rare_result_iou_check/'
    else:
        rare_result_iout_check_dir = args.results_dir + 'result_iou_check/'
    if not os.path.exists(rare_result_iout_check_dir):
        os.mkdir(rare_result_iout_check_dir)
    cv2.imwrite(rare_result_iout_check_dir + image_name, img)


def get_fp_fn_list(cat_ids, img_name, img_id, confusion_matrix, result_list, iou_thres=0.5, score_thres=0.3, whr_thres=3):
    ''' ground truth '''
    good_gt_list = []
    df_lbl = pd.read_csv(args.annos_save_dir + img_name.replace('.jpg', '.txt'), delimiter=' ', header=None)
    df_lbl.iloc[:, 1:] = df_lbl.iloc[:, 1:] * args.input_size
    df_lbl.iloc[:, 1] = df_lbl.iloc[:, 1] - df_lbl.iloc[:, 3]/2
    df_lbl.iloc[:, 2] = df_lbl.iloc[:, 2] - df_lbl.iloc[:, 4]/2
    df_lbl.iloc[:, 3] = df_lbl.iloc[:, 1] + df_lbl.iloc[:, 3]
    df_lbl.iloc[:, 4] = df_lbl.iloc[:, 2] + df_lbl.iloc[:, 4]
    df_lbl = df_lbl.to_numpy()
    for cat_id in cat_ids:
        df_lbl_rare = df_lbl[df_lbl[:, 0] == cat_id, :]
        for dx in range(df_lbl_rare.shape[0]):
            w, h = df_lbl_rare[dx, 3] - df_lbl_rare[dx, 1], df_lbl_rare[dx, 4] - df_lbl_rare[dx, 2]
            whr = np.maximum(w/(h+1e-16), h/(w+1e-16))
            if whr <= whr_thres and w >= 4 and h >= 4:
                good_gt_list.append(df_lbl_rare[dx, :])
    gt_boxes = []
    if good_gt_list:
        good_gt_arr = np.array(good_gt_list)
        gt_boxes = good_gt_arr[:, 1:]
        gt_classes = good_gt_arr[:, 0]

    prd_list = [rx for rx in result_list if rx['image_id'] == img_id]
    prd_lbl_list = []
    for rx in prd_list:
        w, h = rx['bbox'][2:]
        whr = np.maximum(w/(h+1e-16), h/(w+1e-16))
        rx['bbox'][2] = rx['bbox'][2] + rx['bbox'][0]
        rx['bbox'][3] = rx['bbox'][3] + rx['bbox'][1]
        # print('whr', whr)
        if whr <= whr_thres and rx['score'] >= score_thres: #
            prd_lbl = [rx['category_id']]
            prd_lbl.extend([int(b) for b in rx['bbox']]) # xywh
            prd_lbl.extend([rx['score']])
            prd_lbl_list.append(prd_lbl)

    matches = []
    dt_boxes = []
    if prd_lbl_list:
        prd_lbl_arr = np.array(prd_lbl_list)
        # print(prd_lbl_arr.shape)
        dt_scores = prd_lbl_arr[:, -1] # [prd_lbl_arr[:, -1] >= score_thres]
        dt_boxes = prd_lbl_arr[dt_scores >= score_thres][:, 1:-1]
        dt_classes = prd_lbl_arr[dt_scores >= score_thres][:, 0]

    if img_name == '1139_4.jpg':
        print('dt boxes', len(dt_boxes))
        print('gt boxes', len(gt_boxes))
        print(gt_boxes)

    for i in range(len(gt_boxes)):
        for j in range(len(dt_boxes)):
            iou = coord_iou(gt_boxes[i], dt_boxes[j])
            if iou >= iou_thres:
                matches.append([i, j, iou])

    matches = np.array(matches)
    if img_name == '1139_4.jpg':
       print('matches', matches.shape)
    fn_list = []
    fp_list = []

    if matches.shape[0] > 0:
        # Sort list of matches by descending IOU so we can remove duplicate detections
        # while keeping the highest IOU entry.
        matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

        # Remove duplicate detections from the list.
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

        # Sort the list again by descending IOU. Removing duplicates doesn't preserve
        # our previous sort.
        matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

        # Remove duplicate ground truths from the list.
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

    if img_name == '1139_4.jpg':
        print('matches', matches.shape)
        print(matches)
    for i in range(len(gt_boxes)):
        if matches.shape[0] > 0 and matches[matches[:, 0] == i].shape[0] == 1:
            mt_i = matches[matches[:, 0] == i]
            # print('mt_i', mt_i.shape)
            confusion_matrix[cat_ids.index(gt_classes[int(mt_i[0, 0])]), cat_ids.index(dt_classes[int(mt_i[0, 1])])] += 1
        # elif mt_i.shape[0] > 1: # background --> Y_i  FP
        #     drop_match = mt_i[1:]
        #     for n in range(drop_match.shape[0]): # from the second matches, FP
        #         confusion_matrix[-1, cat_ids.index(dt_classes[drop_match[n, 1]])] += 1
        #         c_box_iou = [dt_classes[drop_match[n, 1]]]
        #         c_box_iou.extend(dt_boxes[drop_match[n, 1]])
        #         c_box_iou.append(drop_match[n, 2]) # [cat_id, box[0:4], iou]
        #         fp_list.append(c_box_iou)
        else:
            #fixme
            # unique matches at most has one match for each ground truth
            # 1. ground truth id deleted due to duplicate detections  --> FN
            # 2. matches.shape[0] == 0 --> no matches --> FN
            confusion_matrix[cat_ids.index(gt_classes[i]), -1] += 1
            c_box_iou = [gt_classes[i]]
            c_box_iou.extend(gt_boxes[i])
            c_box_iou.append(0) # [cat_id, box[0:4], iou]
            fn_list.append(c_box_iou)

    for j in range(len(dt_boxes)):
        if img_name == '1139_4.jpg':
            print('j', j)
        #fixme
        # detected object not in the matches --> FP
        # 1. deleted due to duplicate ground truth (background-->Y_prd)
        # 2. lower than iou_thresh (maybe iou=0)  (background-->Y_prd)
        if matches.shape[0] > 0 and matches[matches[:, 1] == j].shape[0] == 0:
            if img_name == '1139_4.jpg':
                print('----------', img_name)
                print('dt matches j =0', j)
            confusion_matrix[-1, cat_ids.index(dt_classes[j])] += 1
            c_box_iou = [dt_classes[j]]
            c_box_iou.extend(dt_boxes[j])
            c_box_iou.append(0) # [cat_id, box[0:4], iou]
            # print(c_box_iou)
            fp_list.append(c_box_iou)
        elif matches.shape[0] > 0 and matches[matches[:, 1] == j].shape[0] == 1:
            if img_name == '1139_4.jpg':
                print('matches==1--j', j)
            continue
        elif matches.shape[0] == 0: #fixme ?
            confusion_matrix[-1, cat_ids.index(dt_classes[j])] += 1
            c_box_iou = [dt_classes[j]]
            c_box_iou.extend(dt_boxes[j])
            c_box_iou.append(0) # [cat_id, box[0:4], iou]
            fp_list.append(c_box_iou)
    return fp_list, fn_list


def get_confusion_matrix(cat_ids, iou_thres=0.5, score_thres=0.3, whr_thres=3):
    '''
    https://github.com/svpino/tf_object_detection_cm/blob/83cb8a1cf3a5abd24b18a5fc79b5ce99e8a9b317/confusion_matrix.py#L37
    '''
    img_ids_names_file = args.txt_save_dir + 'all_image_ids_names_dict_{}cls.json'.format(args.class_num)
    img_ids_names_map = json.load(open(img_ids_names_file))
    img_ids = [int(k) for k in img_ids_names_map.keys()] ## important
    img_names = [v for v in img_ids_names_map.values()]

    cat_id_2_img_names_file = args.txt_save_dir + 'cat_ids_2_val_img_names.json'
    cat_id_2_img_names_json = json.load(open(cat_id_2_img_names_file))
    img_name_list = []
    for rc in cat_ids:
        img_name_list.extend(cat_id_2_img_names_json[str(rc)])
    img_name_list = list(set(img_name_list))
    print('len images', len(img_name_list))
    img_id_list = [img_ids[img_names.index(n)] for n in img_name_list]

    result_json_file = args.results_dir + 'results.json'
    result_allcat_list = json.load(open(result_json_file))
    result_list = []
    # #fixme filter, and rare_result_allcat_list contains rare_cat_ids, rare_img_id_list and object score larger than score_thres
    for ri in result_allcat_list:
        if ri['category_id'] in cat_ids and ri['score'] >= score_thres: # ri['image_id'] in rare_img_id_list and
            result_list.append(ri)
    print('len result_list', len(result_list))
    del result_allcat_list

    confusion_matrix = np.zeros(shape=(len(cat_ids) + 1, len(cat_ids) + 1))

    fn_color = (255, 0, 0) # Blue
    fp_color = (0, 0, 255) # Red
    # save_dir = args.results_dir + 'result_figures_fp_fn_nms_iou{}/'.format(iou_thres)
    save_dir = args.results_dir + 'result_figures_fp_fn_nms_iou{}_cats_{}/'.format(iou_thres, cat_ids)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    confu_mat_dir = args.results_dir + 'result_confusion_matrix_iou{}/'.format(iou_thres)
    if not os.path.exists(confu_mat_dir):
        os.mkdir(confu_mat_dir)
    for ix in range(len(img_id_list)):
        img_id = img_id_list[ix]
    # for img_id in [3160]: # 6121 2609 1238 489 6245
    #     ix = rare_img_id_list.index(img_id)
        img_name = img_name_list[ix]

        #fixme
        fp_list, fn_list = get_fp_fn_list(cat_ids, img_name, img_id, confusion_matrix, result_list, iou_thres, score_thres, whr_thres)

        rcids = []
        img = cv2.imread(args.images_save_dir + img_name)
        for gr in fn_list:
            gr_bx = [int(x) for x in gr[1:-1]]
            img = cv2.rectangle(img, (gr_bx[0], gr_bx[1]), (gr_bx[2], gr_bx[3]), fn_color, 2)
            rcids.append(int(gr[0]))
            cv2.putText(img, text=str(int(gr[0])), org=(gr_bx[2] - 20, gr_bx[3] - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 255))

            for pr in fp_list:
                pr_bx = [int(x) for x in pr[:-1]]
                rcids.append(pr_bx[0])
                img = cv2.rectangle(img, (pr_bx[1], pr_bx[2]), (pr_bx[3], pr_bx[4]), fp_color, 2)
                cv2.putText(img, text='{} {:.3f}'.format(pr_bx[0], pr[-1]), org=(pr_bx[1] + 5, pr_bx[2] + 10), # [pr_bx[0], pr[-1]]
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 0))
            rcids = list(set(rcids))
        cv2.imwrite(save_dir + 'cat{}_'.format(rcids) + img_name_list[ix], img)
    # np.save(confu_mat_dir + 'confusion_matrix.npy', confusion_matrix)

    return confusion_matrix


def plot_img_with_prd_bbx_by_cat_id(cat_id, typestr, score_thres=0.3, whr_thres=3):
    cat_ids_2_img_name_maps = json.load(open(args.txt_save_dir + 'cat_ids_2_{}_img_names.json'.format(typestr)))
    img_names_of_cat_id = cat_ids_2_img_name_maps[cat_id]

    img_ids_2_names = json.load(open(args.txt_save_dir + 'all_image_ids_names_dict_{}cls.json'.format(args.class_num)))
    img_ids = [int(k) for k in img_ids_2_names.keys()]
    img_names = [v for v in img_ids_2_names.values()]

    img_ids_of_cat_id = []
    for n in img_names_of_cat_id:
        ix = img_names.index(n)
        img_ids_of_cat_id.append(img_ids[ix])

    results = json.load(open(args.results_dir + 'results.json'))
    cid = int(cat_id)
    results_cat_id = [rs for rs in results if rs['category_id'] == cid]
    save_dir = args.results_dir + 'results_prd_bbx_cat_id_{}/'.format(cat_id)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for ix, id in enumerate(img_ids_of_cat_id):
        name = img_names_of_cat_id[ix]
        img = cv2.imread(args.images_save_dir + name)
        results_cat_id_img_id = [rs for rs in results_cat_id if rs['image_id'] == id]
        for rs in results_cat_id_img_id:
            if rs['image_id'] == id:
                w, h = rs['bbox'][2:]
                whr = np.maximum(w/(h+1e-16), h/(w+1e-16))
                rs['bbox'][2] = rs['bbox'][2] + rs['bbox'][0]
                rs['bbox'][3] = rs['bbox'][3] + rs['bbox'][1]
                print('whr', whr)
                if whr <= whr_thres and rs['score'] >= score_thres:
                    bbx = [int(x) for x in rs['bbox']]
                    cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[2], bbx[3]), (0, 255, 255), 2)
                    cv2.putText(img, text=cat_id, org=(bbx[0] - 5, bbx[1] - 5), # [pr_bx[0], pr[-1]]
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 0))
        cv2.imwrite(save_dir + 'cat{}_'.format(cat_id) + name, img)


def plot_prd_gt_bbox_by_cat_ids(cat_ids, score_thres=0.3, whr_thres=3):
    '''
    all: all prd & all gt
    '''
    img_ids_names_file = args.txt_save_dir + 'all_image_ids_names_dict_{}cls.json'.format(args.class_num)
    img_ids_names_map = json.load(open(img_ids_names_file))
    img_ids = [int(k) for k in img_ids_names_map.keys()] ## important
    img_names = [v for v in img_ids_names_map.values()]

    df_val_img = pd.read_csv(args.data_save_dir + 'xviewval_img.txt'.format(args.class_num), header=None)
    df_val_gt = pd.read_csv(args.data_save_dir + 'xviewval_lbl.txt'.format(args.class_num), header=None)
    val_img_list = df_val_img[0].tolist()
    val_id_list = df_val_gt[0].tolist()
    val_img_names = [f.split('/')[-1] for f in val_img_list]
    val_img_ids = [img_ids[img_names.index(x)] for x in val_img_names]
    print('gt len', len(val_img_names))

    result_json_file = args.results_dir + 'results.json'
    result_allcat_list = json.load(open(result_json_file))
    result_list = []
    for ri in result_allcat_list:
        if ri['category_id'] in cat_ids and ri['score'] >= score_thres:
            result_list.append(ri)
    print('len result_list', len(result_list))
    del result_allcat_list

    prd_color = (255, 255, 0)
    gt_color = (0, 255, 255) # yellow
    save_dir = args.results_dir + 'result_figures_prd_score_gt_bbox_cat_ids{}/'.format(cat_ids)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for ix in range(len(val_img_list)):
        img = cv2.imread(val_img_list[ix])
        img_size = img.shape[0]
        gt_cat = pd.read_csv(val_id_list[ix], delimiter=' ').to_numpy()
        gt_lbl_list = []

        for gx in range(gt_cat.shape[0]):
            cat_id = int(gt_cat[gx, 0])
            if cat_id in cat_ids:
                gt_cat[gx, 1:] = gt_cat[gx, 1:] * img_size
                gt_cat[gx, 1] = gt_cat[gx, 1] - gt_cat[gx, 3]/2
                gt_cat[gx, 2] = gt_cat[gx, 2] - gt_cat[gx, 4]/2
                w = gt_cat[gx, 3]
                h = gt_cat[gx, 4]
                whr = np.maximum(w/(h+1e-16), h/(w+1e-16))
                if whr > whr_thres or w < 4 or h < 4:
                    continue
                gt_cat[gx, 3] = gt_cat[gx, 3] + gt_cat[gx, 1]
                gt_cat[gx, 4] = gt_cat[gx, 4] + gt_cat[gx, 2]
                gt_lbl_list.append(gt_cat[gx, :])

        print('len gt_lbl_list', len(gt_lbl_list))
        img_prd_lbl_list = []
        for jx in result_list:
            jx_cat_id = jx['category_id']
            if jx['image_id'] == val_img_ids[ix] and jx_cat_id in cat_ids:
                bbx = jx['bbox'] # xtopleft, ytopleft, w, h
                whr = np.maximum(bbx[2]/(bbx[3]+1e-16), bbx[3]/(bbx[2]+1e-16))
                if whr > whr_thres or bbx[2] < 4 or bbx[3] < 4 or jx['score'] < score_thres:
                    continue
                pd_rare = [jx_cat_id]
                bbx[2] = bbx[0] + bbx[2]
                bbx[3] = bbx[1] + bbx[3]
                bbx = [int(b) for b in bbx]
                pd_rare.extend(bbx)
                pd_rare.append(jx['score'])
                img_prd_lbl_list.append(pd_rare)

        for pr in img_prd_lbl_list: # pr_list
            # print(pr)
            bbx = pr[1:-1]
            img = cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[2], bbx[3]), prd_color, 2)
            cv2.putText(img, text='{} {:.3f}'.format(pr[0], pr[-1]), org=(bbx[0] - 5, bbx[1] - 10), # cid, score
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=prd_color)
        for gt in gt_lbl_list: # gt_list
            gt_bbx = gt[1:]
            gt_bbx = [int(g) for g in gt_bbx]
            img = cv2.rectangle(img, (gt_bbx[0], gt_bbx[1]), (gt_bbx[2], gt_bbx[3]), gt_color, 2)
            cv2.putText(img, text=str(int(gt[0])), org=(gt_bbx[2] - 20, gt_bbx[3] - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=gt_color)
        # if gt_list or pr_list:
        if gt_lbl_list or img_prd_lbl_list:
            cv2.imwrite(save_dir + 'cat_{}_'.format(cat_ids) + val_img_names[ix], img)


def summary_confusion_matrix(confusion_matrix, rare_cat_ids, iou_thres=0.5, rare=False):
    '''
    https://github.com/svpino/tf_object_detection_cm/blob/83cb8a1cf3a5abd24b18a5fc79b5ce99e8a9b317/confusion_matrix.py#L37
    '''
    print("\nConfusion Matrix:")
    print(confusion_matrix, "\n")
    results = []
    df_category_id = pd.read_csv(args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num), sep="\t")
    category_names = df_category_id['category'].tolist()
    category_ids = df_category_id['category_id'].tolist()
    rare_cat_names = []
    for ix, id in enumerate(rare_cat_ids):
        name = category_names[category_ids.index(id)]
        rare_cat_names.append(name)

        total_target = np.sum(confusion_matrix[ix, :])
        total_predicted = np.sum(confusion_matrix[:, ix])

        precision = float(confusion_matrix[ix, ix] / total_predicted)
        recall = float(confusion_matrix[ix, ix] / total_target)

        #print('precision_{}@{}IOU: {:.2f}'.format(name, IOU_THRESHOLD, precision))
        #print('recall_{}@{}IOU: {:.2f}'.format(name, IOU_THRESHOLD, recall))

        results.append({'category': name, 'precision_@{}IOU'.format(iou_thres): precision, 'recall_@{}IOU'.format(iou_thres): recall})

    df = pd.DataFrame(results)
    print(df)
    if rare:
        save_dir = args.results_dir + 'rare_result_confusion_matrix_iou{}/'.format(iou_thres)
        file_name = 'rare_result_confusion_mtx.csv'
    else:
        save_dir = args.results_dir + 'result_confusion_matrix_iou{}/'.format(iou_thres)
        file_name = 'result_confusion_mtx.csv'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    df.to_csv(save_dir + file_name)


def plot_confusion_matrix(confusion_matrix, rare_cat_ids, iou_thres=0.5, rare=False):
    df_category_id = pd.read_csv(args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num), sep="\t")
    category_names = df_category_id['category'].tolist()
    category_ids = df_category_id['category_id'].tolist()
    rare_cat_names = []
    for ix, id in enumerate(rare_cat_ids):
        name = category_names[category_ids.index(id)]
        rare_cat_names.append(name)

    rare_cat_names.append('background')
    rare_cat_ids.append(000)
    # print(confusion_matrix.shape)
    df_cm = pd.DataFrame(confusion_matrix[:len(confusion_matrix), :len(confusion_matrix)], index=[i for i in rare_cat_names], columns=[i for i in rare_cat_names])
    plt.figure(figsize=(10, 8))

    p = sn.heatmap(df_cm, annot=True, fmt='g', cmap='YlGnBu')# center=True, cmap=cmap,
    fig = p.get_figure()
    fig.tight_layout()
    plt.xticks(rotation=325)
    if rare:
        save_dir = args.results_dir + 'rare_result_confusion_matrix_iou{}/'.format(iou_thres)
        file_name = 'rare_confusion_matrix.png'
    else:
        save_dir = args.results_dir + 'result_confusion_matrix_iou{}/'.format(iou_thres)
        file_name = 'confusion_matrix.png'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    fig.savefig(save_dir + file_name, bbox_inches='tight')
    plt.show()


def autolabel(ax, rects, x, labels, ylabel, rotation=90, txt_rotation=45):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=txt_rotation)
        if rotation == 0:
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
        else: # rotation=90, multiline label
            xticks = []
            for i in range(len(labels)):
                xticks.append('{} {}'.format(x[i], labels[i]))

            ax.set_xticklabels(xticks, rotation=rotation)
        ax.set_ylabel(ylabel)
        ax.grid(True)


def draw_bar_for_each_cat_cnt_with_txt_subplot(x, ylist, labels, title, save_dir, png_name, rotation=0):
    width = 0.35
    fig, (ax0, ax1, ax2) = plt.subplots(3)

    x0 = x[:20]
    x1 = x[20:40]
    x2 = x[40:60]

    y0 = ylist[:20]
    y1 = ylist[20:40]
    y2 = ylist[40:60]

    ylabel = "Num"

    rects0 = ax0.bar(np.array(x0) - width / 2, y0, width) # , label=labels
    autolabel(ax0, rects0, x0, labels[:20], ylabel, rotation=rotation)

    rects1 = ax1.bar(np.array(x1) - width / 2, y1, width) # , label=labels
    autolabel(ax1, rects1, x1, labels[20:40], ylabel, rotation=rotation)

    rects2 = ax2.bar(np.array(x2) - width / 2, y2, width) # , label=labels
    autolabel(ax2, rects2, x2, labels[40:60], ylabel, rotation=rotation)

    xlabel = 'Categories'
    ax0.set_title(title, literal_eval(args.font3))
    ax2.set_xlabel(xlabel, literal_eval(args.font3))
    # plt.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.savefig(os.path.join(save_dir, png_name))
    plt.show()
    # plt.cla()
    # plt.close()


def draw_bar_for_each_cat_cnt_with_txt_rotation(x, ylist, labels, title, save_dir, png_name, rotation=0, txt_rotation=45, sort=False):
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    fig, ax = plt.subplots(1)
    width = 0.35
    ylabel = "Num"
    if sort:
        rects0 = ax.bar(np.array(range(len(x))) - width / 2, ylist, width) # , label=labels
        ax.set_xticks(range(len(x)))
    else:
        rects0 = ax.bar(np.array(x) - width / 2, ylist, width) # , label=labels
        ax.set_xticks(x)
    autolabel(ax, rects0, x, labels, ylabel, rotation=rotation, txt_rotation=txt_rotation)

    xlabel = 'Categories'
    ax.set_title(title, literal_eval(args.font3))
    ax.set_xlabel(xlabel, literal_eval(args.font3))
    plt.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.savefig(os.path.join(save_dir, png_name))
    plt.show()
    # plt.cla()
    # plt.close()


def draw_bar_for_each_cat_cnt(x, ylist, labels, title, save_dir, png_name, rotation=0):
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    if rotation == 0:
        for i in range(len(ylist)):
            bar = plt.bar(x[i], ylist[i])
        plt.xticks(x, x)
    else: # rotation=90, multiline label
        xticks = []
        for i in range(len(ylist)):
            bar = plt.bar(x[i], ylist[i])
            xticks.append('{} {}'.format(x[i], labels[i]))
        plt.xticks(x, xticks, rotation=rotation)

    xlabel = 'Categories'
    ylabel = "Num"
    # plt.xticks(x, labels, rotation=rotation)
    # plt.xticks(x, x, rotation=rotation)
    plt.title(title, literal_eval(args.font2))
    plt.ylabel(ylabel, literal_eval(args.font2))
    plt.xlabel(xlabel, literal_eval(args.font2))
    plt.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.grid()
    # plt.legend()
    plt.savefig(os.path.join(save_dir, png_name))
    plt.show()
    # plt.cla()


def categories_cnt_for_all():
    df_category_id = pd.read_csv(args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num), sep="\t")
    categories = {}
    for ci in range(args.class_num):
        categories[ci] = 0

    lbl_txt = glob.glob(args.annos_save_dir + '*.txt')
    for i in range(len(lbl_txt)):
        lbl = pd.read_csv(lbl_txt[i], delimiter=' ', header=None)
        for j in range(lbl.shape[0]):
            categories[lbl[0].iloc[j]] += 1

    df_cat = pd.DataFrame({"category_id": [], "category_count": []})
    df_cat['category_id'] = [k for k in categories.keys()]
    df_cat['category_count'] = [v for v in categories.values()]

    df_merge_cat_super = pd.merge(df_cat, df_category_id)
    df_merge_cat_super.to_csv(os.path.join(args.txt_save_dir, 'all_{}_cat_cnt_{}cls.csv'.format(args.input_size, args.class_num)))


def plot_bar_of_each_cat_cnt_with_txt():
    df_category_id_cnt = pd.read_csv(os.path.join(args.txt_save_dir, 'all_{}_cat_cnt_{}cls.csv'.format(args.input_size, args.class_num)), index_col=None)
    y = [v for v in df_category_id_cnt['category_count']]
    x = [k for k in df_category_id_cnt['category_id']]
    label = df_category_id_cnt['category'].to_list()
    title = 'All Chips (608)'
    args.fig_save_dir = args.fig_save_dir + '{}_{}_cls_xcycwh/'.format(args.input_size, args.class_num)
    if not os.path.exists(args.fig_save_dir):
        os.makedirs(args.fig_save_dir)

    # png_name_txt = 'cat_count_all_{}_with_txt_subplot.png'.format(args.input_size)
    # draw_bar_for_each_cat_cnt_with_txt_subplot(x, y, label, title, args.fig_save_dir, png_name_txt, rotation=300)
    png_name_txt = 'cat_count_all_{}_with_txt_rotation.png'.format(args.input_size)
    draw_bar_for_each_cat_cnt_with_txt_rotation(x, y, label, title, args.fig_save_dir, png_name_txt, rotation=300, txt_rotation=45)
    # png_name = 'cat_count_all_original.png'
    # draw_bar_for_each_cat_cnt(x, y, label, title, args.fig_save_dir, png_name, rotation=0)


def draw_rare_cat(N):

    title = 'Categories Count Least Rare {}'.format(N)
    save_dir = args.fig_save_dir + '{}_{}_cls_xcycwh/'.format(args.input_size, args.class_num)
    png_name = 'all_{}_cat_cnt_with_top{}_least_rare.png'.format(args.input_size, N)
    df_cat = pd.read_csv(os.path.join(args.txt_save_dir, 'all_{}_cat_cnt_{}cls.csv'.format(args.input_size, args.class_num)))
    x_ids = df_cat['category_id']
    x_cats = df_cat['category']
    y = df_cat['category_count']

    y_sort_inx = np.argsort(y)[:N]
    labels = x_cats[y_sort_inx].to_list()
    print(y_sort_inx)
    # x = [x_ids[i] for i in y_sort_inx]
    x = x_ids[y_sort_inx].to_list()
    y = y[y_sort_inx].to_list()
    print('category ids', x)
    print('category', labels)
    print('num', y)

    draw_bar_for_each_cat_cnt_with_txt_rotation(x, y, labels, title, save_dir, png_name, rotation=80, sort=True)


def get_cat_names_by_cat_ids(cat_ids):
    img_ids_names_file = args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num, args.class_num)
    df_cat_id_names = pd.read_csv(img_ids_names_file, delimiter='\t')
    category_names = df_cat_id_names['category'].tolist()
    category_ids = df_cat_id_names['category_id'].tolist()
    result_cat_names = []
    for c in cat_ids:
        result_cat_names.append(category_names[category_ids.index(c)])
    return result_cat_names



'''
Datasets
https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data
_aug: Augmented dataset
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str,
                        help="Path to folder containing image chips (ie 'Image_Chips/') ",
                        default='/media/lab/Yang/data/xView/train_images/')

    parser.add_argument("--json_filepath", type=str, help="Filepath to GEOJSON coordinate file",
                        default='/media/lab/Yang/data/xView/xView_train.geojson')

    parser.add_argument("--xview_yolo_dir", type=str, help="dir to xViewYOLO",
                        default='/media/lab/Yang/data/xView_YOLO/')

    parser.add_argument("--images_save_dir", type=str, help="to save chip trn val images files",
                        default='/media/lab/Yang/data/xView_YOLO/images/')

    parser.add_argument("--txt_save_dir", type=str, help="to save  related label files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')

    parser.add_argument("--annos_save_dir", type=str, help="to save txt annotation files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')

    parser.add_argument("--ori_lbl_dir", type=str, help="to save original labels files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/original/')

    parser.add_argument("--fig_save_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/figures/')

    parser.add_argument("--data_save_dir", type=str, help="to save data files",
                        default='../data_xview/')

    parser.add_argument("--results_dir", type=str, help="to save category files",
                        default='../result_output/{}_cls/')

    parser.add_argument("--cat_sample_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/')

    parser.add_argument("--img_bbx_figures_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/{}/{}_cls/img_name_2_bbx_figures/')

    parser.add_argument("--cat_bbx_patches_dir", type=str, help="to split cats bbx patches",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/{}/{}_cls/cat_split_patches/')
    parser.add_argument("--cat_bbx_origins_dir", type=str, help="to split cats bbx patches",
                        default='/media/lab/Yang/data/xView_YOLO/labels/original/{}_cls/cat_split_origins/')

    parser.add_argument("--val_percent", type=float, default=0.20,
                        help="Percent to split into validation (ie .25 = val set is 25% total)")
    parser.add_argument("--font3", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 10}")
    parser.add_argument("-ft2", "--font2", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 13}")
    parser.add_argument("--class_num", type=int, default=6, help="Number of Total Categories")  # 60  6
    parser.add_argument("--seed", type=int, default=1024, help="random seed")
    parser.add_argument("--input_size", type=int, default=608, help="Number of Total Categories")  # 300 416

    parser.add_argument("--rare", type=bool, default=False, help="Number of Total Categories") # -----------==--------change

    args = parser.parse_args()

    args.images_save_dir = args.images_save_dir + '{}/'.format(args.input_size)
    args.annos_save_dir = args.annos_save_dir + '{}/{}_cls_xcycwh/'.format(args.input_size, args.class_num)
    args.txt_save_dir = args.txt_save_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)
    args.cat_sample_dir = args.cat_sample_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)
    args.ori_lbl_dir = args.ori_lbl_dir + '{}_cls/'.format(args.class_num)
    args.data_save_dir = args.data_save_dir + '{}_cls/'.format(args.class_num)
    args.img_bbx_figures_dir = args.img_bbx_figures_dir.format(args.input_size, args.class_num)
    args.results_dir = args.results_dir.format(args.class_num)

    if not os.path.exists(args.txt_save_dir):
        os.makedirs(args.txt_save_dir)

    if not os.path.exists(args.annos_save_dir):
        os.makedirs(args.annos_save_dir)

    if not os.path.exists(args.images_save_dir):
        os.makedirs(args.images_save_dir)

    if not os.path.exists(args.cat_sample_dir):
        os.makedirs(args.cat_sample_dir)

    if not os.path.exists(args.ori_lbl_dir):
        os.makedirs(args.ori_lbl_dir)

    if not os.path.exists(args.data_save_dir):
        os.makedirs(args.data_save_dir)

    if not os.path.exists(args.img_bbx_figures_dir):
        os.makedirs(args.img_bbx_figures_dir)
    args.cat_bbx_patches_dir = args.cat_bbx_patches_dir.format(args.input_size, args.class_num)
    args.cat_bbx_origins_dir = args.cat_bbx_origins_dir.format(args.class_num)
    if not os.path.exists(args.cat_bbx_patches_dir):
        os.makedirs(args.cat_bbx_patches_dir)
    if not os.path.exists(args.cat_bbx_origins_dir):
        os.makedirs(args.cat_bbx_origins_dir)

    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger(__name__)

    '''
    create chips and label txt and get all images json, convert from *.geojson to *.json
    
    get all images json, convert from *.geojson to *.json
    # convert_geojson_to_json()
    create label txt files
    # create_label_txt()
    
    '''
    # create_chips_and_txt_geojson_2_json()

    '''
    create xview.names 
    '''
    # create_xview_names()

    '''
    split train:val  randomly split chips
    default split
    '''
    # split_trn_val_with_chips()

    '''
    split train:val  randomly split tifs
    '''
    # split_trn_val_with_tifs()

    '''
    create json for val according to all jsons 
    '''
    # create_json_for_val_according_to_all_json()

    '''
    create xview.data includes cls_num, trn, val, ...
    '''
    # create_xview_data()

    '''
    create json for train or val according to all jsons 
    '''
    # typestr = 'train'
    # typestr = 'val'
    # create_json_for_train_or_val_according_to_all_json(typestr)

    '''
    categories count for all label files
    and plot category count
    '''
    # categories_cnt_for_all()
    # plot_bar_of_each_cat_cnt_with_txt()

    '''
    cats_split_crop_bbx_from_patches for val images
    '''
    # typestr = 'val'
    # whr_thres = 3
    # N = 200 # each category select N samples
    # cats_split_crop_bbx_from_patches(typestr, whr_thres, N)

    '''
    cats_split_crop_bbx_from_origins for val images
    '''
    # whr_thres = 3
    # imgN = 50 # show imgN images
    # N = 20 # each image select N samples
    # cats_split_crop_bbx_from_origins(whr_thres, imgN, N)
    #
    # cat_ids = np.arange(0, 6).tolist()
    # cid_wh_maps = json.load(open(os.path.join(args.ori_lbl_dir, 'xView_cid_2_wh_maps.json')))
    # draw_wh_scatter_for_cats(cid_wh_maps, cat_ids)

    # img_name = '525.tif'
    # plot_image_with_gt_bbx_by_image_name_from_origin(img_name)

    '''
    IoU check by image name
    '''
    # score_thres = 0.3
    # iou_thres = 0.5
    # # image_name = '310_12.jpg'
    # # image_name = '1090_3.jpg'
    # image_name = '1139_4.jpg'
    # check_prd_gt_iou(image_name, score_thres, iou_thres, args.rare)

    '''
    get cat ids to img names maps
    '''
    # cat_ids = ['0', '1', '2', '3', '4', '5']
    # # # cat_ids = np.arange(0, 6).tolist()
    # # typestr='train'
    # typestr='val'
    # get_train_or_val_imgs_by_cat_id(cat_ids, typestr)

    '''
    val results confusion matrix val results statistic FP FN NMS
    '''
    cat_ids = np.arange(0, 6).tolist()
    cat_ids = [0]
    score_thres = 0.3
    whr_thres = 3
    iou_thres = 0.5
    args.rare = False
    confusion_matrix = get_confusion_matrix(cat_ids, iou_thres, score_thres, whr_thres)
    # summary_confusion_matrix(confusion_matrix, cat_ids, iou_thres, args.rare)

    # cat_ids = np.arange(0, 6).tolist()
    # iou_thres = 0.5
    # confusion_matrix = np.load(args.results_dir + 'result_confusion_matrix_iou{}/confusion_matrix.npy'.format(iou_thres))
    # plot_confusion_matrix(confusion_matrix, cat_ids, iou_thres, args.rare)

    '''
    draw cat cnt least N rare classes
    '''
    # N = 20
    # draw_rare_cat(N)

    '''
    plot img with prd bbox by cat id
    '''
    # score_thres = 0.3
    # whr_thres = 3
    # typestr='val'
    # cat_id = '0'
    # plot_img_with_prd_bbx_by_cat_id(cat_id, typestr, score_thres, whr_thres)

    '''
    plot prd bbox with object score and gt bbox
    '''
    # cat_ids = np.arange(0, 6).tolist()
    # cat_ids = [0]
    # score_thres = 0.3
    # iou_thres=0.5
    # whr_thres=3
    # plot_prd_gt_bbox_by_cat_ids(cat_ids, score_thres, whr_thres)

    '''
    get prd bbx by image name and cat id
    '''
    # score_thres = 0.3
    # whr_thres = 3
    # cat_ids = [0]
    # image_name = '1139_4.jpg'
    # plot_img_with_prd_bbx_by_image_name(image_name, cat_ids, score_thres, whr_thres)

    '''
    plot img with gt bbx by img name from patches
    '''
    # image_name = '1139_4.jpg'
    # image_name = '5_7.jpg'
    # # image_name = '2230_3.jpg'
    # image_name = '525_20.jpg'
    # image_name = '1139_4.jpg'
    # plot_val_image_with_gt_bbx_by_image_name_from_patches(image_name)

    '''
    get img id by image_name
    '''
    # image_name = '2230_3.jpg' # 6121
    # image_name = '145_16.jpg' # 2609
    # image_name = '1154_6.jpg' # 1238
    # image_name = '1076_16.jpg' # 489
    # image_name = '2294_5.jpg' #6245
    # image_name = '1568_0.jpg' # 3160
    # image_name='1139_4.jpg' # 1124
    # image_id = get_img_id_by_image_name(image_name)
    # print(image_id)

    '''
    IoU check by image name
    '''
    # score_thres = 0.5
    # iou_thres = 0.5
    # image_name = '310_12.jpg'
    # args.rare = False
    # check_prd_gt_iou(image_name, score_thres, iou_thres, args.rare)


    '''
    get class names by cat_id
    '''
    # cat_ids = np.arange(0, 6).tolist()
    # cat_names = get_cat_names_by_cat_ids(cat_ids)
    # print(rare_cat_names)

    '''
    crop patches per bbox from 608x608 patches
    '''
    # cat_ids = np.arange(0, 6).tolist()
    # cat_names = ['Aircraft', 'Motor Vehicles', 'Railway', 'Maritime', 'Construction Vehicles', 'Fixed Facilities']
    # # typestr = 'train'
    # typestr = 'val'
    # cats_split_crop_bbx_from_patches(cat_ids, cat_names, typestr)

    '''
    crop patches per bbox from tiles 
    '''
    # cat_ids = np.arange(0, 6).tolist()
    # rare_cat_names = ['Fixed-Wing Aircraft', 'Straddle Carrier', 'Helicopter', 'Small Aircraft', 'Passenger/Cargo Plane', 'Yacht']
    # rare_cats_split_crop_bbx_from_tiles(rare_cat_ids, rare_cat_names)
    #
    # clabel_wh_maps = json.load(open(os.path.join(args.ori_lbl_dir, 'xView_rare_clabel_2_wh_maps.json')))
    # draw_wh_scatter_for_rare_cats(clabel_wh_maps, rare_cat_ids, rare_cat_names)

