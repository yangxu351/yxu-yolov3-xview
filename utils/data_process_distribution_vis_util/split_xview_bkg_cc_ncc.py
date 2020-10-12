import os
import argparse
import glob
import shutil
import numpy as np
import pandas as pd


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def split_trn_val_for_each_CC_step_by_step(cc_id, seed, txt_name, base_pxwhrs):
    '''
    split CC1, CC2 separately
    1. split CC* to train and val
    2. split Non-CC to train and val
    3. split BKG to train and val, #BKG  == CC* + Non-CC
    :return:
    '''
    all_ori_dir = args.annos_save_dir[:-1] + '_with_id0'
    all_cc_dir =  args.annos_save_dir[:-1] + '_cc{}_with_id'.format(cc_id)
    all_bkg_dir = args.annos_save_dir[:-1] + '_bkg'
    cc_img_dir = args.images_save_dir[:-1] + '_cc{}'.format(cc_id)
    bkg_img_dir = args.images_save_dir[:-1] + '_noairplane_bkg_chips'
    img_dir = args.images_save_dir

    all_ori_files = glob.glob(os.path.join(all_ori_dir, '*.txt'))
    all_cc_files = glob.glob(os.path.join(all_cc_dir, '*.txt'))
    all_cc_names = [os.path.basename(f) for f in all_cc_files]
    all_ncc_files = [f for f in all_ori_files if is_non_zero_file(f) and os.path.basename(f) not in all_cc_names]
    all_ept_files = [f for f in all_ori_files if not is_non_zero_file(f)]
    all_bkg_files = glob.glob(os.path.join(all_bkg_dir, '*.txt'))

    np.random.seed(seed)
    val_cc_num = round(len(all_cc_files)*args.val_percent)
    trn_cc_num = len(all_cc_files) - val_cc_num

    val_ncc_num = round(len(all_ncc_files)*args.val_percent)
    trn_ncc_num = len(all_ncc_files) - val_ncc_num

    trn_ept_num = round(len(all_ept_files)*args.val_percent)
    val_ept_num = len(all_ept_files) - trn_ept_num

    val_bkg_num = val_cc_num + val_ncc_num - trn_ept_num
    trn_bkg_num = trn_cc_num + trn_ncc_num - val_ept_num

    all_cc_files = np.random.permutation(all_cc_files)
    trn_cc_files = all_cc_files[:trn_cc_num]
    val_cc_files = all_cc_files[trn_cc_num:]

    all_ncc_files = np.random.permutation(all_ncc_files)
    trn_ncc_files = all_ncc_files[:trn_ncc_num]
    val_ncc_files = all_ncc_files[trn_ncc_num:]

    all_ept_files = np.random.permutation(all_ept_files)
    trn_ept_files = all_ept_files[:trn_ept_num]
    val_ept_files = all_ept_files[val_ept_num:]

    all_bkg_files = np.random.permutation(all_bkg_files)
    trn_bkg_files = all_bkg_files[:trn_bkg_num]
    val_bkg_files = all_bkg_files[trn_bkg_num:trn_bkg_num+val_bkg_num]

    data_save_dir =  os.path.join(args.data_save_dir, base_pxwhrs, 'CC')
    if not os.path.exists(data_save_dir):
        os.mkdir(data_save_dir)

    trn_cc_img_txt = open(os.path.join(data_save_dir, 'only_cc{}_trn_img_{}.txt'.format(cc_id, base_pxwhrs)), 'w')
    trn_cc_lbl_txt = open(os.path.join(data_save_dir, 'only_cc{}_trn_lbl_{}.txt'.format(cc_id, base_pxwhrs)), 'w')
    trn_nccbkg_img_txt = open(os.path.join(data_save_dir, '{}_trn_img_{}.txt'.format(txt_name, base_pxwhrs)), 'w')
    trn_nccbkg_lbl_txt = open(os.path.join(data_save_dir, '{}_trn_lbl_{}.txt'.format(txt_name, base_pxwhrs)), 'w')

    nccbkg_img_trn_dir = args.images_save_dir[:-1] + '_nccbkg_trn_cc{}'.format(cc_id)
    if not os.path.exists(nccbkg_img_trn_dir):
        os.mkdir(nccbkg_img_trn_dir)
    nccbkg_lbl_trn_dir = args.annos_save_dir[:-1] + '_nccbkg_trn_cc{}_with_id'.format(cc_id)
    if not os.path.exists(nccbkg_lbl_trn_dir):
        os.mkdir(nccbkg_lbl_trn_dir)

    for f in trn_cc_files:
        trn_cc_lbl_txt.write("%s\n" % f)
        img_name = os.path.basename(f).replace('.txt', '.jpg')
        trn_cc_img_txt.write("%s\n" % os.path.join(cc_img_dir, img_name))
    for f in trn_ncc_files:
        trn_nccbkg_lbl_txt.write("%s\n" % f)
        img_name = os.path.basename(f).replace('.txt', '.jpg')
        trn_nccbkg_img_txt.write("%s\n" % os.path.join(img_dir, img_name))
        shutil.copy(f, os.path.join(nccbkg_lbl_trn_dir, os.path.basename(f)))
        shutil.copy(os.path.join(img_dir, img_name), os.path.join(nccbkg_img_trn_dir, img_name))

    for f in trn_ept_files:
        trn_nccbkg_lbl_txt.write("%s\n" % f)
        img_name = os.path.basename(f).replace('.txt', '.jpg')
        trn_nccbkg_img_txt.write("%s\n" % os.path.join(img_dir, img_name))
        shutil.copy(f, os.path.join(nccbkg_lbl_trn_dir, os.path.basename(f)))
        shutil.copy(os.path.join(img_dir, img_name), os.path.join(nccbkg_img_trn_dir, img_name))

    for f in trn_bkg_files:
        trn_nccbkg_lbl_txt.write("%s\n" % f)
        img_name = os.path.basename(f).replace('.txt', '.jpg')
        trn_nccbkg_img_txt.write("%s\n" % os.path.join(bkg_img_dir, img_name))
        shutil.copy(f, os.path.join(nccbkg_lbl_trn_dir, os.path.basename(f)))
        shutil.copy(os.path.join(bkg_img_dir, img_name), os.path.join(nccbkg_img_trn_dir, img_name))

    trn_nccbkg_lbl_txt.close()
    trn_nccbkg_img_txt.close()

    ###### validate xview_cc_nrc_bkg
    nccbkg_img_val_dir = args.images_save_dir[:-1] + '_nccbkg_val_cc{}'.format(cc_id)
    if not os.path.exists(nccbkg_img_val_dir):
        os.mkdir(nccbkg_img_val_dir)
    nccbkg_lbl_val_dir = args.annos_save_dir[:-1] + '_nccbkg_val_cc{}_with_id'.format(cc_id)
    if not os.path.exists(nccbkg_lbl_val_dir):
        os.mkdir(nccbkg_lbl_val_dir)
    cc_img_val_dir = args.images_save_dir[:-1] + '_val_cc{}'.format(cc_id)
    if not os.path.exists(cc_img_val_dir):
        os.mkdir(cc_img_val_dir)
    cc_lbl_val_dir = args.annos_save_dir[:-1] + '_val_cc{}_with_id'.format(cc_id)
    if not os.path.exists(cc_lbl_val_dir):
        os.mkdir(cc_lbl_val_dir)
    # val_cc_img_txt = open(os.path.join(data_save_dir, 'only_cc{}_val_img_{}.txt'.format(cc_id, base_pxwhrs)), 'w')
    # val_cc_lbl_txt = open(os.path.join(data_save_dir, 'only_cc{}_val_lbl_{}.txt'.format(cc_id, base_pxwhrs)), 'w')
    # val_nccbkg_img_txt = open(os.path.join(data_save_dir, '{}_val_img_{}.txt'.format(txt_name, base_pxwhrs)), 'w')
    # val_nccbkg_lbl_txt = open(os.path.join(data_save_dir, '{}_val_lbl_{}.txt'.format(txt_name, base_pxwhrs)), 'w')
    val_nccbkg_cc_img_txt = open(os.path.join(data_save_dir, 'xview_nccbkg_cc{}_val_img_{}.txt'.format(cc_id, base_pxwhrs)), 'w')
    val_nccbkg_cc_lbl_txt = open(os.path.join(data_save_dir, 'xview_nccbkg_cc{}_val_lbl_{}.txt'.format(cc_id, base_pxwhrs)), 'w')

    for f in val_cc_files:
        val_nccbkg_cc_lbl_txt.write("%s\n" % f)
        img_name = os.path.basename(f).replace('.txt', '.jpg')
        val_nccbkg_cc_img_txt.write("%s\n" % os.path.join(cc_img_dir, img_name))
        shutil.copy(f, os.path.join(cc_lbl_val_dir, os.path.basename(f)))
        shutil.copy(os.path.join(cc_img_dir, img_name), os.path.join(cc_img_val_dir, img_name))

    for f in val_ncc_files:
        val_nccbkg_cc_lbl_txt.write("%s\n" % f)
        img_name = os.path.basename(f).replace('.txt', '.jpg')
        val_nccbkg_cc_img_txt.write("%s\n" % os.path.join(img_dir, img_name))
        shutil.copy(f, os.path.join(nccbkg_lbl_val_dir, os.path.basename(f)))
        shutil.copy(os.path.join(img_dir, img_name), os.path.join(nccbkg_img_val_dir, img_name))

    for f in val_ept_files:
        val_nccbkg_cc_lbl_txt.write("%s\n" % f)
        img_name = os.path.basename(f).replace('.txt', '.jpg')
        val_nccbkg_cc_img_txt.write("%s\n" % os.path.join(img_dir, img_name))
        shutil.copy(f, os.path.join(nccbkg_lbl_val_dir, os.path.basename(f)))
        shutil.copy(os.path.join(img_dir, img_name), os.path.join(nccbkg_img_val_dir, img_name))

    for f in val_bkg_files:
        val_nccbkg_cc_lbl_txt.write("%s\n" % f)
        img_name = os.path.basename(f).replace('.txt', '.jpg')
        val_nccbkg_cc_img_txt.write("%s\n" % os.path.join(bkg_img_dir, img_name))
        shutil.copy(f, os.path.join(nccbkg_lbl_val_dir, os.path.basename(f)))
        shutil.copy(os.path.join(bkg_img_dir, img_name), os.path.join(nccbkg_img_val_dir, img_name))

    val_nccbkg_cc_lbl_txt.close()
    val_nccbkg_cc_img_txt.close()


def create_xview_cc_nccbkg_data(cc_id, data_name, txt_name, seed=17, val_aug=False):
    base_cmt = 'px{}whr{}_seed{}'.format(px_thres, whr_thres, seed)
    data_save_dir =  os.path.join(args.data_save_dir, base_cmt, 'CC')
    print('data_save_dir', data_save_dir)
    data_txt = open(os.path.join(data_save_dir, '{}_{}.data'.format(data_name, base_cmt)), 'w')
    data_txt.write(
        'xview_train={}\n'.format(os.path.join(data_save_dir, '{}_trn_img_{}.txt'.format(txt_name, base_cmt))))
    data_txt.write(
        'xview_train_label={}\n'.format(os.path.join(data_save_dir, '{}_trn_lbl_{}.txt'.format(txt_name, base_cmt))))

    data_txt.write(
        'cc_train={}\n'.format(os.path.join(data_save_dir,  'only_cc{}_trn_img_{}.txt'.format(cc_id, base_cmt))))
    data_txt.write(
        'cc_train_label={}\n'.format(os.path.join(data_save_dir, 'only_cc{}_trn_lbl_{}.txt'.format(cc_id, base_cmt))))
    if val_aug:
        data_txt.write(
            'valid={}\n'.format(os.path.join(data_save_dir, 'xview_nccbkg_aug_cc{}_test_img_{}.txt'.format(cc_id, base_cmt))))
        data_txt.write(
            'valid_label={}\n'.format(os.path.join(data_save_dir, 'xview_nccbkg_aug_cc{}_test_lbl_{}.txt'.format(cc_id, base_cmt))))
    else:
        data_txt.write(
            'valid={}\n'.format(os.path.join(data_save_dir, '{}_val_img_{}.txt'.format(data_name, base_cmt))))
        data_txt.write(
            'valid_label={}\n'.format(os.path.join(data_save_dir, '{}_val_lbl_{}.txt'.format(data_name, base_cmt))))

    df_trn_nccbkg = pd.read_csv(os.path.join(data_save_dir, '{}_trn_img_{}.txt'.format(txt_name, base_cmt)), header=None)
    df_trn_cc = pd.read_csv(os.path.join(data_save_dir,  'only_cc{}_trn_img_{}.txt'.format(cc_id, base_cmt)), header=None)
    xview_trn_num = df_trn_nccbkg.shape[0] + df_trn_cc.shape[0]

    data_txt.write('xview_number={}\n'.format(xview_trn_num))
    data_txt.write('classes=%s\n' % str(args.class_num))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval=color\n')
    data_txt.close()


def create_aug_cc_nccbkg_test(cc_id):
    pxwhrs = 'px23whr3_seed17'
    eht = 'easy'
    base_dir = os.path.join(args.data_save_dir, pxwhrs, 'CC')
    test_lbl_txt = open(os.path.join(base_dir, 'xview_nccbkg_aug_cc{}_test_lbl_{}.txt'.format(cc_id, pxwhrs)), 'w')
    test_img_txt = open(os.path.join(base_dir, 'xview_nccbkg_aug_cc{}_test_img_{}.txt'.format(cc_id, pxwhrs)), 'w')
    lbl_dir = args.annos_save_dir[:-1] + '_val_cc{}_with_id_aug'.format(cc_id)
    img_dir = args.images_save_dir[:-1] + '_val_cc{}_aug'.format(cc_id)
    aug_rc_lbls = np.sort(glob.glob(os.path.join(lbl_dir, '*.txt')))
    for f in aug_rc_lbls:
        test_lbl_txt.write('%s\n' % f)
        name = os.path.basename(f).replace('.txt', '.jpg')
        test_img_txt.write('%s\n' % os.path.join(img_dir, name))

    nccbkg_img_path = args.images_save_dir[:-1] + '_nccbkg_val_cc{}'.format(cc_id)
    nccbkg_lbl_dir = args.annos_save_dir[:-1] + '_nccbkg_val_cc{}_with_id'.format(cc_id)
    nccbkg_lbl_files = glob.glob(os.path.join(nccbkg_lbl_dir, '*.txt'))

    for f in nccbkg_lbl_files:
        test_lbl_txt.write('%s\n' % f)
        name = os.path.basename(f).replace('.txt', '.jpg')
        test_img_txt.write('%s\n' % os.path.join(nccbkg_img_path, name))
    test_img_txt.close()
    test_lbl_txt.close()

    data_txt = open(os.path.join(base_dir, 'xview_nccbkg_aug_cc{}_test_{}.data'.format(cc_id, pxwhrs)), 'w')
    data_txt.write('classes=%s\n' % str(args.class_num))
    data_txt.write('test=./data_xview/{}_cls/{}/CC/xview_nccbkg_aug_cc{}_test_img_{}.txt\n'.format(args.class_num, pxwhrs, cc_id, pxwhrs))
    data_txt.write('test_label=./data_xview/{}_cls/{}/CC/xview_nccbkg_aug_cc{}_test_lbl_{}.txt\n'.format(args.class_num, pxwhrs, cc_id, pxwhrs))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(args.class_num))
    data_txt.close()


def get_args(px_thres=None, whr_thres=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--xview_yolo_dir", type=str, help="dir to xViewYOLO",
                        default='/media/lab/Yang/data/xView_YOLO/')

    parser.add_argument("--images_save_dir", type=str, help="to save chip trn val images files",
                        default='/media/lab/Yang/data/xView_YOLO/images/')

    # parser.add_argument("--images_save_dir", type=str, help="to save chip trn val images files",
    #                     default='/media/lab/Yang/data/xView_YOLO/images/{}_')

    parser.add_argument("--syn_save_dir", type=str, help="",
                        default='/media/lab/Yang/data/synthetic_data/')

    parser.add_argument("--txt_save_dir", type=str, help="to save  related label files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')

    parser.add_argument("--annos_save_dir", type=str, help="to save txt annotation files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')

    parser.add_argument("--data_save_dir", type=str, help="to save data files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_cls/')
    parser.add_argument("--cat_sample_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/')
    parser.add_argument("--val_percent", type=float, default=0.2,
                        help="0.24 0.2 Percent to split into validation (ie .25 = val set is 25% total)")
    parser.add_argument("--font3", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 10}")
    parser.add_argument("-ft2", "--font2", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 13}")
    parser.add_argument("--class_num", type=int, default=1, help="Number of Total Categories")  # 60  6 1
    parser.add_argument("--seed", type=int, default=17, help="random seed") #fixme ---- 1024 17
    parser.add_argument("--input_size", type=int, default=608, help="Number of Total Categories")  # 300 416

    args = parser.parse_args()
    #fixme ----------==--------change
    args.images_save_dir = args.images_save_dir + '{}_{}cls/'.format(args.input_size, args.class_num)
    args.annos_save_dir = args.annos_save_dir + '{}/{}_cls_xcycwh_px{}whr{}/'.format(args.input_size, args.class_num, px_thres, whr_thres)
    args.txt_save_dir = args.txt_save_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)
    args.data_save_dir = args.data_save_dir.format(args.class_num)
    if not os.path.exists(args.txt_save_dir):
        os.makedirs(args.txt_save_dir)

    if not os.path.exists(args.annos_save_dir):
        os.makedirs(args.annos_save_dir)

    if not os.path.exists(args.images_save_dir):
        os.makedirs(args.images_save_dir)

    if not os.path.exists(args.data_save_dir):
        os.makedirs(args.data_save_dir)

    args.cat_sample_dir = args.cat_sample_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)
    if not os.path.exists(args.cat_sample_dir):
        os.makedirs(args.cat_sample_dir)
    return args


if __name__ == '__main__':
    px_thres=23
    whr_thres=3
    args = get_args(px_thres, whr_thres)

    '''
    backup *lbl_with_id
    lable all bbx with label 0
    '''
    # ori_dir = args.annos_save_dir[:-1] + '_with_id'
    # ori0_dir = args.annos_save_dir[:-1] + '_with_id0'
    # if not os.path.exists(ori0_dir):
    #     os.mkdir(ori0_dir)
    # ori_files = glob.glob(os.path.join(ori_dir, '*.txt'))
    # for f in ori_files:
    #     if is_non_zero_file(f):
    #         df = pd.read_csv(f, header=None, sep=' ')
    #         df.loc[:, 5] = 0
    #         df.to_csv(os.path.join(ori0_dir, os.path.basename(f)), header=False, index=False, sep=' ')
    #     else:
    #         shutil.copy(f, os.path.join(ori0_dir, os.path.basename(f)))

    '''
    split CC1, CC2 separately
    1. split CC* to train and val
    2. split Non-CC to train and val
    3. split BKG to train and val, #BKG  == CC* + Non-CC
    '''
    # seed = 17
    # base_pxwhrs = 'px{}whr{}_seed{}'.format(px_thres, whr_thres, seed)
    # # cc_id = 1
    # cc_id = 2
    # txt_name = 'xview_nccbkg_nocc{}'.format(cc_id)
    # split_trn_val_for_each_CC_step_by_step(cc_id, seed, txt_name, base_pxwhrs)
    # data_name = 'xview_nccbkg_cc{}'.format(cc_id)
    # create_xview_cc_nccbkg_data(cc_id, data_name, txt_name, seed)

    '''
    augment CC 
    left-right flip
    rotate 90, 180, 270
    '''
    # from utils.xview_synthetic_util.resize_flip_rotate_images import flip_rotate_images, flip_rotate_coordinates
    # # cc_id = 1
    # cc_id = 2
    # img_dir = args.images_save_dir[:-1] + '_val_cc{}'.format(cc_id)
    # img_list = glob.glob(os.path.join(img_dir, '*.jpg'))
    # print('img_list', len(img_list))
    # for f in img_list:
    #     name = os.path.basename(f)
    #     img_dir = os.path.dirname(f)
    #     flip_rotate_images(img_name=name, src_dir=img_dir)
    #
    # img_dir = args.images_save_dir[:-1] + '_val_cc{}_aug'.format(cc_id)
    # lbl_dir = args.annos_save_dir[:-1] + '_val_cc{}_with_id'.format(cc_id)
    # save_dir = args.cat_sample_dir + 'image_with_bbox/val_cc{}_aug/'.format(cc_id)
    # lbl_list = glob.glob(os.path.join(lbl_dir, '*.txt'))
    # for lbl in lbl_list:
    #     name = os.path.basename(lbl).split('.')[0]
    #     angle_list = [90, 180, 270]
    #     for angle in angle_list:
    #         flip_rotate_coordinates(img_dir, lbl_dir, save_dir, name, angle=angle)
    #     flip_list = ['lr'] # 'tb
    #     for flip in flip_list:
    #         flip_rotate_coordinates(img_dir, lbl_dir, save_dir, name, flip=flip)

    '''
    create *.data with val augmented CC  
    '''
    seed = 17
    cc_id = 1
    # cc_id = 2
    txt_name = 'xview_nccbkg_nocc{}'.format(cc_id)
    data_name = 'xview_nccbkg_aug_cc{}'.format(cc_id)
    create_xview_cc_nccbkg_data(cc_id, data_name, txt_name, seed, val_aug=True)

    '''
    create test data of Aug CC
    '''
    # cc_id = 1
    # # cc_id = 2
    # create_aug_cc_nccbkg_test(cc_id)
