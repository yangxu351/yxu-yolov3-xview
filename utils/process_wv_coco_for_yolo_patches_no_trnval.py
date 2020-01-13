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
import pandas as pd
from ast import literal_eval
import json
import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm
import shutil

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

    txt_norm_dir = os.path.join(args.txt_save_dir, '{}_cls_xcycwh'.format(args.class_num))
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
            f_txt = open(os.path.join(txt_norm_dir, ana_txt_name), 'w')
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


# def convert_geojson_to_json():
#
#     dfs = pd.read_csv(args.txt_save_dir + 'image_names_{}_{}cls.csv'.format(args.input_size, args.class_num))
#     ids = dfs['id'].astype(np.int64)
#     names = dfs['file_name'].to_list()
#     num_files = len(names)
#     trn_annotation_list = []
#     trn_image_info_list = []
#     coords, chips, classes, features_ids = wv.get_labels(args.json_filepath, args.class_num)
#
#     images_save_dir = args.images_save_dir
#
#     for i in ids:
#         img_name = names[i]
#         img = wv.get_image(images_save_dir + img_name)
#
#         image_info = {
#             "id": ids[i],
#             "file_name": img_name,
#             "height": img.shape[0],
#             "width": img.shape[1],
#             "date_captured": datetime.datetime.utcnow().isoformat(' '),
#             "license": 1,
#             "coco_url": "",
#             "flickr_url": ""
#         }
#         trn_image_info_list.append(image_info)
#
#         box_ids = features_ids[chips == img_name]
#         box = coords[chips == img_name]
#         classes_final = classes[chips == img_name]
#         for d in range(box_ids.shape[0]):
#             # create annotation_info
#             bbx = box[d]
#             bbx[0] = max(0, bbx[0])
#             bbx[1] = max(0, bbx[1])
#             bbx[2] = min(bbx[2], img.shape[1])
#             bbx[3] = min(bbx[3], img.shape[0])
#             annotation_info = {
#                 "id": box_ids[d],
#                 "image_id": ids[i],
#                 # "image_name": img_name, #fixme: there aren't 'image_name'
#                 "category_id": np.int(classes_final[d]),
#                 "iscrowd": 0,
#                 "area": (bbx[2] - bbx[0] + 1) * (bbx[3] - bbx[1] + 1),
#                 "bbox": [bbx[0], bbx[1], bbx[2] - bbx[0], bbx[3] - bbx[1]], # (w1, h1, w, h)
#                 "segmentation": [],
#             }
#             trn_annotation_list.append(annotation_info)
#
#     # trn_img_txt.close()
#     # trn_lbl_txt.close()
#
#     trn_instance = {'info': 'xView Train yx185 created', 'license': ['license'], 'images': trn_image_info_list,
#                     'annotations': trn_annotation_list, 'categories': wv.get_all_categories(args.class_num)}
#     json_file = os.path.join(args.txt_save_dir, 'xViewall_{}_{}cls_xtlytlwh.json'.format(args.input_size, args.class_num)) # topleft
#     json.dump(trn_instance, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)

# def create_label_txt():
#     coords, chips, classes, features_ids = wv.get_labels(args.json_filepath, args.class_num)
#     # gs = json.load(open('/media/lab/Yang/data/xView/xView_train.geojson'))
#
#     img_save_dir = os.path.join(args.images_save_dir)
#     if not os.path.exists(img_save_dir):
#         os.makedirs(img_save_dir)
#     txt_save_dir = os.path.join(args.txt_save_dir, '{}_cls'.format(args.class_num))
#     if not os.path.exists(txt_save_dir):
#         os.makedirs(txt_save_dir)
#
#     img_files = glob.glob(args.image_folder + '*.tif')
#
#     for im in img_files:
#         im_name = im.split('/')[-1]
#         '''
#          # drop 1395.tif
#         '''
#         if im_name == '1395.tif':
#             continue
#         #fixme note
#         img = wv.get_image(im) # h, w, c
#         ht = img.shape[0]
#         wd = img.shape[1]
#
#         # w, h = img.shape[:2]
#         txt_name = im_name.replace('.tif', '.txt')
#         f_txt = open(os.path.join(txt_save_dir, txt_name), 'w')
#         im_cls = classes[chips==im_name]
#         im_coords = coords[chips==im_name]
#         for i in range(len(im_cls)):
#             # bbox = literal_eval(im_coords[i])
#             bbox = im_coords[i]
#             bbox[0] = max(0, bbox[0])
#             bbox[2] = min(bbox[2], wd)
#
#             bbox[1] = max(0, bbox[1])
#             bbox[3] = min(bbox[3], ht)
#
#             bbx = convert_norm(img.shape[:2], bbox)
#             f_txt.write("%s %s %s %s %s\n" % (
#                 np.int(im_cls[i]), bbx[0], bbx[1], bbx[2], bbx[3]))
#         f_txt.close()


def create_xview_names():
    df_cat = pd.read_csv(args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num), delimiter='\t')
    cat_names = df_cat['category'].to_list()
    f_txt = open(os.path.join(args.data_save_dir, 'xview.names'), 'w')
    for i in range(len(cat_names)):
        f_txt.write("%s\n" % cat_names[i])
    f_txt.close()


def split_trn_val_with_chips():

    all_files = glob.glob(args.images_save_dir + '*.jpg')
    lbl_path = args.txt_save_dir + '{}_cls_xcycwh/'.format(args.class_num)
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

    args.data_save_dir = args.data_save_dir + '{}_cls'.format(args.class_num)
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
    lbl_path = args.txt_save_dir + '{}_cls_xcycwh/'.format(args.class_num)
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

    args.data_save_dir = args.data_save_dir + '{}_cls'.format(args.class_num)
    shutil.copyfile(os.path.join(args.txt_save_dir, 'xviewtrain_img_tifsplit.txt'), os.path.join(args.data_save_dir, 'xviewtrain_img_tifsplit.txt'))
    shutil.copyfile(os.path.join(args.txt_save_dir, 'xviewtrain_lbl_tifsplit.txt'), os.path.join(args.data_save_dir, 'xviewtrain_lbl_tifsplit.txt'))
    shutil.copyfile(os.path.join(args.txt_save_dir, 'xviewval_img_tifsplit.txt'), os.path.join(args.data_save_dir, 'xviewval_img_tifsplit.txt'))
    shutil.copyfile(os.path.join(args.txt_save_dir, 'xviewval_lbl_tifsplit.txt'), os.path.join(args.data_save_dir, 'xviewval_lbl_tifsplit.txt'))


def create_json_for_val_according_to_all_json():
    json_all_file = os.path.join(args.txt_save_dir, 'xViewall_{}_{}cls_xtlytlwh.json'.format(args.input_size, args.class_num)) # topleft
    all_json = json.load(open(json_all_file))
    all_imgs_info = all_json['images']
    all_annos_info = all_json['annotations']
    all_cats_info = all_json['categories']

    all_img_id_maps = json.load(open(args.txt_save_dir + 'all_image_ids_names_dict_{}cls.json'.format(args.class_num)))
    all_imgs = [i for i in all_img_id_maps.values()]
    all_img_ids = [int(i) for i in all_img_id_maps.keys()]

    val_img_files = pd.read_csv(os.path.join(args.data_save_dir, 'xviewval_img.txt'), header=None)
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
    json_file = os.path.join(args.txt_save_dir, 'xViewval_{}_{}cls_xtlytlwh.json'.format(args.input_size, args.class_num)) # topleft
    json.dump(trn_instance, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)


def rare_cat_id_to_imgs(rare_cat_ids, typestr='all'):
    cat_img_ids_maps = json.load(open(os.path.join(args.label_save_dir, '{}_cat_img_ids_dict_{}cls.json'.format(typestr, args.class_num))))
    img_ids_names_maps = json.load(open(os.path.join(args.label_save_dir, '{}_image_ids_names_dict_{}cls.json'.format(typestr, args.class_num))))

    val_rare_img_txt = open(os.path.join(args.data_save_dir, 'xviewval_rare_img.txt'), 'w')
    val_rare_lbl_txt = open(os.path.join(args.data_save_dir, 'xviewval_rare_lbl.txt'), 'w')
    img_path = args.images_save_dir
    lbl_path = args.label_save_dir + '{}_cls_xcycwh/'.format(args.class_num)

    val_files = pd.read_csv(os.path.join(args.data_save_dir, 'xviewval_img.txt'), header=None)
    val_img_names = [f.split('/')[-1] for f in val_files[0]]
    rare_cat_val_imgs = {}
    for rc in rare_cat_ids:
        cat_img_ids = cat_img_ids_maps[rc]
        rare_cat_val_imgs[rc] = []
        rare_cat_all_imgs_files = []
        for c in cat_img_ids:
            rare_cat_all_imgs_files.append(img_ids_names_maps[str(c)]) # rare cat id map to all imgs
        rare_cat_val_imgs[rc] = [v for v in rare_cat_all_imgs_files if v in val_img_names] # rare cat id map to val imgs
        print('cat imgs in val', rare_cat_val_imgs[rc])
        print('cat {} total imgs {}, val imgs {}'.format(rc, len(rare_cat_all_imgs_files), len(rare_cat_val_imgs)))
        for rimg in rare_cat_val_imgs[rc]:
            val_rare_img_txt.write("%s\n" % (img_path + rimg))
            val_rare_lbl_txt.write("%s\n" % (lbl_path + rimg.replace('.jpg', '.txt')))

    rare_cat_val_imgs_files = os.path.join(args.label_save_dir, 'rare_cat_ids_2_val_imgs_files.json')
    json.dump(rare_cat_val_imgs, open(rare_cat_val_imgs_files, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)


def create_json_for_val_rare_according_to_all_json():
    json_all_file = os.path.join(args.txt_save_dir, 'xViewall_{}_{}cls_xtlytlwh.json'.format(args.input_size, args.class_num)) # topleft
    all_json = json.load(open(json_all_file))
    all_imgs_info = all_json['images']
    all_annos_info = all_json['annotations']
    all_cats_info = all_json['categories']

    all_img_id_maps = json.load(open(args.txt_save_dir + 'all_image_ids_names_dict_{}cls.json'.format(args.class_num)))
    all_imgs = [i for i in all_img_id_maps.values()]
    all_img_ids = [int(i) for i in all_img_id_maps.keys()]

    val_img_files = pd.read_csv(os.path.join(args.data_save_dir, 'xviewval_rare_img.txt'), header=None)
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

    trn_instance = {'info': 'xView val rare chips 600 yx185 created', 'license': ['license'], 'images': image_info_list,
                    'annotations': anno_info_list, 'categories': all_cats_info}
    json_file = os.path.join(args.txt_save_dir, 'xViewval_rare_{}_{}cls_xtlytlwh.json'.format(args.input_size, args.class_num)) # topleft
    json.dump(trn_instance, open(json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)


def create_xview_data():
    data_txt = open(os.path.join(args.data_save_dir, 'xview.data'), 'w')
    data_txt.write('classes=%s\n' % str(args.class_num))
    data_txt.write('train=./data_xview/{}_cls/xviewtrain_img.txt\n'.format(args.class_num))
    data_txt.write('train_label=./data_xview/{}_cls/xviewtrain_lbl.txt\n'.format(args.class_num))
    data_txt.write('valid=./data_xview/{}_cls/xviewval_img.txt\n'.format(args.class_num))
    data_txt.write('valid_label=./data_xview/{}_cls/xviewval_lbl.txt\n'.format(args.class_num))
    data_txt.write('valid_rare=./data_xview/{}_cls/xviewval_rare_img.txt\n'.format(args.class_num))
    data_txt.write('valid_rare_lbl=./data_xview/{}_cls/xviewval_rare_lbl.txt\n'.format(args.class_num))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval=xview')
    data_txt.close()


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
    # plt.rcParams['figure.figsize'] = (10.0, 8.0)
    fig, (ax0, ax1, ax2) = plt.subplots(3)
    width = 0.35
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
    plt.cla()


def categories_cnt_for_all():
    df_category_id = pd.read_csv(args.data_save_dir + 'categories_id_color_diverse_{}.txt'.format(args.class_num), sep="\t")
    categories = {}
    for ci in range(args.class_num):
        categories[ci] = 0

    lbl_txt = glob.glob(args.txt_save_dir + '{}_cls_xcycwh/'.format(args.class_num) + '*.txt')
    for i in range(len(lbl_txt)):
        lbl = pd.read_csv(lbl_txt[i], delimiter=' ', header=None)
        for j in range(lbl.shape[0]):
            categories[lbl[0].iloc[j]] += 1

    df_cat = pd.DataFrame({"category_id": [], "category_count": []})
    df_cat['category_id'] = [k for k in categories.keys()]
    df_cat['category_count'] = [v for v in categories.values()]

    df_merge_cat_super = pd.merge(df_category_id, df_cat)
    df_merge_cat_super.to_csv(os.path.join(args.txt_save_dir, 'all_{}_cat_cnt_{}cls.csv'.format(args.input_size, args.class_num)))

    y = [v for v in categories.values()]
    x = [k for k in categories.keys()]
    label = df_category_id['category'].to_list()
    title = 'All Chips (608)'
    args.fig_save_dir = args.fig_save_dir + '{}_{}_cls_xcycwh/'.format(args.input_size, args.class_num)
    if not os.path.exists(args.fig_save_dir):
        os.makedirs(args.fig_save_dir)

    png_name_txt = 'cat_count_all_{}_with_txt_subplot.png'.format(args.input_size)
    draw_bar_for_each_cat_cnt_with_txt_subplot(x, y, label, title, args.fig_save_dir, png_name_txt, rotation=300)
    # png_name_txt = 'cat_count_all_{}_with_txt_rotation.png'.format(args.input_size)
    # draw_bar_for_each_cat_cnt_with_txt_rotation(x, y, label, title, args.fig_save_dir, png_name_txt, rotation=270, txt_rotation=90)
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

    parser.add_argument("--txt_save_dir", type=str, help="to save txt labels files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')

    parser.add_argument("--fig_save_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/figures/')

    parser.add_argument("--data_save_dir", type=str, help="to save data files",
                        default='../data_xview/')

    # parser.add_argument("--cat_dir", type=str, help="to save category files",
    #                     default='../data_xview/')

    parser.add_argument("--cat_sample_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/')

    parser.add_argument("--val_percent", type=float, default=0.20,
                        help="Percent to split into validation (ie .25 = val set is 25% total)")
    parser.add_argument("--font3", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 10}")

    parser.add_argument("--class_num", type=int, default=60, help="Number of Total Categories")  # 60  6
    parser.add_argument("--seed", type=int, default=1024, help="random seed")
    parser.add_argument("--input_size", type=int, default=608, help="Number of Total Categories")  # 300 416

    args = parser.parse_args()

    args.txt_save_dir = args.txt_save_dir + '{}/'.format(args.input_size)
    args.images_save_dir = args.images_save_dir + '{}/'.format(args.input_size)
    args.cat_sample_dir = args.cat_sample_dir + '{}/'.format(args.input_size)
    args.data_save_dir = args.data_save_dir + '{}_cls/'.format(args.class_num)

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
    rare classes random split check
    '''
    # cat_ids = ['18', '46', '23', '33', '59']
    # typestr='all'
    # # cat_id = '18'
    # rare_cat_id_to_imgs(cat_ids, typestr)

    '''
    create json for val according to all jsons 
    '''
    # create_json_for_val_rare_according_to_all_json()

    '''
    create xview.data includes cls_num, trn, val, ...
    '''
    # create_xview_data()

    '''
    categories count for all label files
    '''
    # categories_cnt_for_all()

    '''
    draw cat cnt least N rare classes
    '''
    # N = 20
    # draw_rare_cat(N)



    # import json
    # import cv2
    # gs = json.load(open('/media/lab/Yang/data/xView/xView_train.geojson'))
    # feas = gs['features']
    # for i in range(len(feas)):
    #     img_name = feas[i]['properties']['image_id']
    #     img = cv2.imread(args.image_path + img_name) # h,w,c
    #     ht = img.shape[0]
    #     wd = img.shape[1]
    #
    #     coords = literal_eval(feas[i]['properties']['bounds_imcoords'])
    #     coords[0] = max(0, coords[0])
    #     coords[2] = min(coords[2], wd)
    #
    #     coords[1] = max(0, coords[1])
    #     coords[3] = min(coords[3], ht)
