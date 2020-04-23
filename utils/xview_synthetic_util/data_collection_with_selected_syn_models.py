import glob
import numpy as np
import os
import pandas as pd
import shutil
from utils.data_process_distribution_vis_util import process_wv_coco_for_yolo_patches_no_trnval as pwv
from utils.xview_synthetic_util import preprocess_synthetic_data_distribution as pps


def generate_new_xview_lbl_with_model_id(type='val', nonmatch=3, comments='', px_thres=None, whr_thres=None):
    '''
    generate new xview train val annotations (with model id)
    generate new xview train val list
    :return:
    '''
    args = pwv.get_args(px_thres, whr_thres)
    if comments:
        lbl_name = 'xview{}_lbl_{}.txt'.format(type, comments)
        xview_val_lbl_with_model_name = 'xview{}_lbl_{}_with_model.txt'.format(type, comments)
        data_save_dir = os.path.join(args.data_save_dir, comments)
        data_backup_dir = os.path.join(args.data_list_save_dir, comments)
        if not os.path.exists(data_backup_dir):
            os.mkdir(data_backup_dir)
    else:
        lbl_name = 'xview{}_lbl.txt'.format(type)
        xview_val_lbl_with_model_name = 'xview{}_lbl_with_model.txt'.format(type)
        data_save_dir = args.data_save_dir
        data_backup_dir = args.data_list_save_dir

    ori_val_lbl_txt = os.path.join(data_save_dir, lbl_name)
    df_ori_val = pd.read_csv(ori_val_lbl_txt, header=None)
    ori_val_names = [os.path.basename(f) for f in df_ori_val.iloc[:, 0]]

    xview_val_lbl_with_model_txt = open(os.path.join(args.data_save_dir, xview_val_lbl_with_model_name), 'w')
    if px_thres:
        des_val_lbl_path = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px{}whr{}_only_model/'.format(px_thres, whr_thres)
        src_lbl_path = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px{}whr{}_model/'.format(px_thres, whr_thres)
    else:
        des_val_lbl_path = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_only_model/'
        src_lbl_path = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_model/'

    if not os.path.exists(des_val_lbl_path):
        os.mkdir(des_val_lbl_path)

    for i in range(len(ori_val_names)):
        f = os.path.join(src_lbl_path, ori_val_names[i])
        name = os.path.basename(f)
        if not pps.is_non_zero_file(f):
            shutil.copy(f, os.path.join(des_val_lbl_path, name))
            xview_val_lbl_with_model_txt.write('%s\n' % os.path.join(des_val_lbl_path, name))
            continue

        colomns = np.arange(0, 6)
        src_lbl = pd.read_csv(f, header=None, sep=' ', index_col=False, names=colomns)
        des_lbl_txt = open(os.path.join(des_val_lbl_path, name), 'w')

        for i in range(src_lbl.shape[0]):
            if np.isnan(src_lbl.iloc[i, 5]):
                modelid = nonmatch
            else:
                modelid = src_lbl.iloc[i, 5]
            des_lbl_txt.write("%d %.8f %.8f %.8f %.8f %d\n" % (src_lbl.iloc[i, 0], src_lbl.iloc[i, 1], src_lbl.iloc[i, 2], src_lbl.iloc[i, 3], src_lbl.iloc[i, 4], modelid))
        des_lbl_txt.close()
        xview_val_lbl_with_model_txt.write('%s\n' % os.path.join(des_val_lbl_path, name))
    xview_val_lbl_with_model_txt.close()
    shutil.copy(os.path.join(data_save_dir, xview_val_lbl_with_model_name), os.path.join(data_backup_dir, xview_val_lbl_with_model_name))


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
        src_lbl = pd.read_csv(src_lbl_file, header=None, sep=' ', index_col=False, names=colomns) #fixme --- **** index_col=False ****
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



def change_labels_from_px6whr4_to_px20whr4():
    '''
    drop b-boxes whose length is larger than px_thres
    or drop b-boxes whose wh_ratio is larger than whr_thres
    :return:
    '''
    syn_args = pps.get_syn_args()
    px_thres = 20
    whr_thres = 4
    label_path = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px20whr4_only_model/'

    if not os.path.exists(label_path):
        os.mkdir(label_path)
    else:
        shutil.rmtree(label_path)
        os.mkdir(label_path)
    src_files = glob.glob('/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_only_model/*.txt')
    for f in src_files:
        shutil.copy(f, label_path)
    label_files = np.sort(glob.glob(os.path.join(label_path, '*.txt')))
    for lf in label_files:
        if not pps.is_non_zero_file(lf):
            continue
        df_txt = pd.read_csv(lf, header=None, delimiter=' ')
        # print('before ', df_txt.shape)
        # if '1945_5.txt' in lf:
        #     print(lf)
        for ix in df_txt.index:
            df_txt.loc[ix, 1:4] = df_txt.loc[ix, 1:4] * syn_args.tile_size
            # whr = np.max(df_txt.loc[ix, 3] / df_txt.loc[ix, 4], df_txt.loc[ix, 4] / df_txt.loc[ix, 3])
            whr = np.maximum(df_txt.loc[ix, 3] / df_txt.loc[ix, 4], df_txt.loc[ix, 4] / df_txt.loc[ix, 3])
            if int(df_txt.loc[ix, 3]) <= px_thres or int(df_txt.loc[ix, 4]) <= px_thres or whr > whr_thres:
                df_txt = df_txt.drop(ix)
            else:
                df_txt.loc[ix, 1:4] = df_txt.loc[ix, 1:4]/syn_args.tile_size
        df_txt.to_csv(lf, header=False, index=False, sep=' ')

    a = glob.glob('/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px20whr4/*.txt')
    b = glob.glob('/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px20whr4_only_model/*.txt')

    aa = []
    bb = []
    for i in range(len(a)):
        if not pps.is_non_zero_file(a[i]):
            aa.append(os.path.basename(a[i]))
        if not pps.is_non_zero_file(b[i]):
            bb.append(os.path.basename(b[i]))
    ax = [x for x in aa if x not in bb]
    bx = [x for x in bb if x not in aa]
    print(ax)
    print(bx)


def change_labels_from_px6whr4_to_px23whr4():
    '''
    only drop b-boxes those at the edge of the image, and the length of bbox is larger than px_thres
    or drop b-boxes whose wh_ratio is larger than whr_thres
    :return:
    '''
    syn_args = pps.get_syn_args()
    px_thres = 23
    whr_thres = 4
    label_path = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px{}whr{}_only_model/'.format(px_thres, whr_thres)

    if not os.path.exists(label_path):
        os.mkdir(label_path)
    else:
        shutil.rmtree(label_path)
        os.mkdir(label_path)

    src_files = glob.glob('/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_only_model/*.txt')
    for f in src_files:
        shutil.copy(f, label_path)
    label_files = np.sort(glob.glob(os.path.join(label_path, '*.txt')))
    for lf in label_files:
        if not pps.is_non_zero_file(lf):
            continue
        df_txt = pd.read_csv(lf, header=None, delimiter=' ')
        # print('before ', df_txt.shape)
        # if '1044_1.txt' in lf:
        #     print(lf)
        for ix in df_txt.index:
            #fixme
            # df_txt.loc[ix, 1:4] = df_txt.loc[ix, 1:4] * syn_args.tile_size
            # # whr = np.max(df_txt.loc[ix, 3] / df_txt.loc[ix, 4], df_txt.loc[ix, 4] / df_txt.loc[ix, 3])
            # whr = np.maximum(df_txt.loc[ix, 3] / df_txt.loc[ix, 4], df_txt.loc[ix, 4] / df_txt.loc[ix, 3])
            # if int(df_txt.loc[ix, 3]) <= px_thres or int(df_txt.loc[ix, 4]) <= px_thres or whr > whr_thres:
            #     df_txt = df_txt.drop(ix)
            # else:
            #     df_txt.loc[ix, 1:4] = df_txt.loc[ix, 1:4]/syn_args.tile_size
            bbx = df_txt.loc[ix, 1:4] * syn_args.tile_size
            xl = round(bbx.loc[1] - bbx.loc[3]/2)
            yl = round(bbx.loc[2] - bbx.loc[4]/2)
            xr = round(bbx.loc[1] + bbx.loc[3]/2)
            yr = round(bbx.loc[2] + bbx.loc[4]/2)
            bbx_wh = max(bbx.loc[3]/bbx.loc[4], bbx.loc[4]/bbx.loc[3])
            w = round(bbx.loc[3])
            h = round(bbx.loc[4])
            if bbx_wh >= whr_thres:
                df_txt = df_txt.drop(ix)
            elif (xl<=0 or xr>=syn_args.tile_size-1) and (w <= px_thres or h <= px_thres):
                df_txt = df_txt.drop(ix)
            elif (yl<=0 or yr>=syn_args.tile_size-1) and (w <= px_thres or h <= px_thres):
                df_txt = df_txt.drop(ix)
        df_txt.to_csv(lf, header=False, index=False, sep=' ')

    a = glob.glob('/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px{}whr{}/*.txt'.format(px_thres, whr_thres))
    b = glob.glob('/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px{}whr{}_only_model/*.txt'.format(px_thres, whr_thres))

    aa = []
    bb = []
    for i in range(len(a)):
        if not pps.is_non_zero_file(a[i]):
            aa.append(os.path.basename(a[i]))
        if not pps.is_non_zero_file(b[i]):
            bb.append(os.path.basename(b[i]))
    ax = [x for x in aa if x not in bb]
    bx = [x for x in bb if x not in aa]
    print(ax)
    print(bx)


if __name__ == '__main__':
    '''
    generate new xviewval_lbl_with_model.txt
    '''
    # type = 'train'
    # type = 'val'
    # nonmatch = 3
    # generate_new_xview_lbl_with_model_id(type, nonmatch)

    # type = 'val'
    # nonmatch = 5
    # px_thres = 23
    # whr_thres = 3
    # comments = 'px{}whr{}_seed17'.format(px_thres, whr_thres)
    # generate_new_xview_lbl_with_model_id(type, nonmatch, comments, px_thres, whr_thres)


    '''
    generate new syn_*_lbl_with_model.txt
    '''
    # comments = '_with_model'
    # nonmatch = 3
    # generate_new_syn_lbl_with_model_id(comments, nonmatch)


    '''
    change labels from px6whr4 to px20whr4
    '''
    # change_labels_from_px6whr4_to_px20whr4()

    # ['1585_7.txt', '1945_5.txt']
    # for bbx in ax:
    #     df_a = pd.read_csv(os.path.join('/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px20whr4_only_model/', bbx), header=None, delimiter=' ').to_numpy()
    #     df_a[:, 1:5] = df_a[:,1:5]*syn_args.tile_size
    #     print(df_a)

    # df_txt = pd.read_csv('/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_only_model/1945_5.txt', header=None, sep=' ')
    # df_txt.iloc[:, 1:5] = df_txt.iloc[:, 1:5] * syn_args.tile_size
    # df_txt = df_txt[df_txt[:, 4]>20]

    '''
    change labels from px6whr4 to px23whr4
    '''
    # change_labels_from_px6whr4_to_px23whr4()
