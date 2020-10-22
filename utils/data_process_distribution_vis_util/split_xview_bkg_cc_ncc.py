import os
import argparse
import glob
import shutil
import numpy as np
import pandas as pd


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

def move_ept_to_bkg():
    args = get_args(px_thres=23, whr_thres=3)
    lbl_path = args.annos_save_dir[:-1]
    lbl_model_path = args.annos_save_dir[:-1] + '_with_id'
    lbl_model0_path = args.annos_save_dir[:-1] + '_with_id0'
    lbl_model5_path = args.annos_save_dir[:-1] + '_with_model'
    lbl_rcid_path = args.annos_save_dir[:-1] + '_with_rcid'
    bkg_lbl_dir = args.annos_save_dir[:-1] + '_bkg'

    images_save_dir = args.images_save_dir
    bkg_img_dir = args.images_save_dir[:-1] + '_noairplane_bkg_chips'

    lbl_files = glob.glob(os.path.join(lbl_path, '*.txt'))
    lbl_ept_files = [f for f in lbl_files if not is_non_zero_file(f)]
    print('len of lbl_ept_files', len(lbl_ept_files))
    txt_ept_names = open(os.path.join(args.xview_yolo_dir, 'removed_ept_files_from_lbl.txt'), 'w')
    for f in lbl_ept_files:
        lbl_name = os.path.basename(f)
        txt_ept_names.write('%s\n' % lbl_name)
        img_name = lbl_name.replace('.txt', '.jpg')
        shutil.move(f, os.path.join(bkg_lbl_dir, lbl_name))
        shutil.move(os.path.join(images_save_dir, img_name), os.path.join(bkg_img_dir, img_name))
        if os.path.exists(os.path.join(lbl_model_path, lbl_name)):
            os.remove(os.path.join(lbl_model_path, lbl_name))
        if os.path.exists(os.path.join(lbl_model0_path, lbl_name)):
            os.remove(os.path.join(lbl_model0_path, lbl_name))
        if os.path.exists(os.path.join(lbl_model5_path, lbl_name)):
            os.remove(os.path.join(lbl_model5_path, lbl_name))
        if os.path.exists(os.path.join(lbl_rcid_path, lbl_name)):
            os.remove(os.path.join(lbl_rcid_path, lbl_name))
    txt_ept_names.close()
    bkg_files = glob.glob(os.path.join(bkg_lbl_dir, '*.txt'))
    print('len of bkg_files', len(bkg_files))


def split_bkg_into_train_val(comments='px{}whr{}_seed{}', seed=17, px_thres=23, whr_thres=3):
    '''
    split data contains no aircrafts (bkg images)
    '''
    comments = comments.format(px_thres, whr_thres, seed)
    args = get_args(px_thres, whr_thres)
    bkg_lbl_dir = args.annos_save_dir[:-1] + '_bkg'
    bkg_img_dir = args.images_save_dir[:-1] + '_noairplane_bkg_chips'
    data_save_dir = args.data_save_dir
    if comments:
        txt_save_dir = args.data_list_save_dir + comments # + '_bh'+ '/'
        if not os.path.exists(txt_save_dir):
            os.makedirs(txt_save_dir)
        data_save_dir = os.path.join(data_save_dir, comments)
        if not os.path.exists(data_save_dir):
            os.makedirs(data_save_dir)
    else:
        txt_save_dir = args.data_list_save_dir

    ##### images that contain no aircrafts-- BKG
    bkg_lbl_files = glob.glob(os.path.join(bkg_lbl_dir, '*.txt'))
    print('bkg_lbl_files', len(bkg_lbl_files))
    bkg_lbl_files.sort()
    bkg_img_files = [os.path.join(bkg_img_dir, os.path.basename(f).replace('.txt', '.jpg')) for f in bkg_lbl_files]

    np.random.seed(seed)
    bkg_ixes = np.random.permutation(len(bkg_lbl_files))
    print('bkg_ixes[:10]', bkg_ixes[:10])
    val_bkg_num = int(len(bkg_lbl_files)*args.val_percent)
    trn_bkg_num = len(bkg_lbl_files) - val_bkg_num
    trn_bkg_lbl_files =[bkg_lbl_files[i] for i in bkg_ixes[:trn_bkg_num ]]
    val_bkg_lbl_files = [bkg_lbl_files[i] for i in bkg_ixes[trn_bkg_num: trn_bkg_num + val_bkg_num]]
    trn_bkg_img_files = [bkg_img_files[i] for i in bkg_ixes[: trn_bkg_num]]
    val_bkg_img_files = [bkg_img_files[i] for i in bkg_ixes[trn_bkg_num: trn_bkg_num + val_bkg_num]]
    print('trn_bkg_lbl_files', len(trn_bkg_lbl_files), len(trn_bkg_img_files))
    print('val_bkg_img_files', len(val_bkg_lbl_files), len(val_bkg_img_files))

    ###### train mixed batch of xview_rc + xview_nrc_bkg
    trn_bkg_img_txt = open(os.path.join(txt_save_dir, 'xview_bkg_train_img_{}.txt'.format(comments)), 'w')
    trn_bkg_lbl_txt = open(os.path.join(txt_save_dir, 'xview_bkg_train_lbl_{}.txt'.format(comments)), 'w')

    ###### validate xview_rc_nrc_bkg
    val_bkg_img_txt = open(os.path.join(txt_save_dir, 'xview_bkg_val_img_{}.txt'.format(comments)), 'w')
    val_bkg_lbl_txt = open(os.path.join(txt_save_dir, 'xview_bkg_val_lbl_{}.txt'.format(comments)), 'w')

    trn_bkg_lbl_dir = args.annos_save_dir[:-1] + '_trn_bkg_lbl'
    val_bkg_lbl_dir = args.annos_save_dir[:-1] + '_val_bkg_lbl'
    if os.path.exists(trn_bkg_lbl_dir):
        shutil.rmtree(trn_bkg_lbl_dir)
        os.mkdir(trn_bkg_lbl_dir)
    else:
        os.mkdir(trn_bkg_lbl_dir)
    if os.path.exists(val_bkg_lbl_dir):
        shutil.rmtree(val_bkg_lbl_dir)
        os.mkdir(val_bkg_lbl_dir)
    else:
        os.mkdir(val_bkg_lbl_dir)

    trn_bkg_img_dir = args.images_save_dir[:-1] + '_trn_bkg_img'
    val_bkg_img_dir = args.images_save_dir[:-1] + '_val_bkg_img'
    if os.path.exists(trn_bkg_img_dir):
        shutil.rmtree(trn_bkg_img_dir)
        os.mkdir(trn_bkg_img_dir)
    else:
        os.mkdir(trn_bkg_img_dir)

    if os.path.exists(val_bkg_img_dir):
        shutil.rmtree(val_bkg_img_dir)
        os.mkdir(val_bkg_img_dir)
    else:
        os.mkdir(val_bkg_img_dir)

    for i in range(len(trn_bkg_lbl_files)):
        trn_bkg_lbl_txt.write("%s\n" % trn_bkg_lbl_files[i])
        trn_bkg_img_txt.write("%s\n" % trn_bkg_img_files[i])
        lbl_name = os.path.basename(trn_bkg_lbl_files[i])
        img_name = lbl_name.replace('.txt', '.jpg')
        shutil.copy(os.path.join(bkg_lbl_dir, lbl_name), os.path.join(trn_bkg_lbl_dir, lbl_name))
        shutil.copy(os.path.join(bkg_img_dir, img_name), os.path.join(trn_bkg_img_dir, img_name))
    trn_bkg_lbl_txt.close()
    trn_bkg_img_txt.close()

    for j in range(len(val_bkg_lbl_files)):
        val_bkg_lbl_txt.write("%s\n" % val_bkg_lbl_files[j])
        val_bkg_img_txt.write("%s\n" % val_bkg_img_files[j])
        lbl_name = os.path.basename(val_bkg_lbl_files[j])
        img_name = lbl_name.replace('.txt', '.jpg')
        shutil.copy(os.path.join(bkg_lbl_dir, lbl_name), os.path.join(val_bkg_lbl_dir, lbl_name))
        shutil.copy(os.path.join(bkg_img_dir, img_name), os.path.join(val_bkg_img_dir, img_name))
    val_bkg_lbl_txt.close()
    val_bkg_img_txt.close()

    shutil.copy(os.path.join(txt_save_dir, 'xview_bkg_train_img_{}.txt'.format(comments)),
                    os.path.join(data_save_dir, 'xview_bkg_train_img_{}.txt'.format(comments)))
    shutil.copy(os.path.join(txt_save_dir, 'xview_bkg_val_img_{}.txt'.format(comments)),
                    os.path.join(data_save_dir, 'xview_bkg_val_img_{}.txt'.format(comments)))


def split_trn_val_for_each_CC_step_by_step(cc_id, seed, txt_name, base_pxwhrs):
    '''
    split CC1, CC2 separately
    1. split CC* to train and val
    2. split Non-CC to train and val
    3. combine BKG of train and val
    :return:
    '''
    all_ori_dir = args.annos_save_dir[:-1] + '_with_id0'
    all_cc_dir =  args.annos_save_dir[:-1] + '_cc{}_with_id'.format(cc_id)
    val_bkg_lbl_dir = args.annos_save_dir[:-1] + '_val_bkg_lbl'
    trn_bkg_lbl_dir = args.annos_save_dir[:-1] + '_trn_bkg_lbl'
    val_bkg_img_dir = args.images_save_dir[:-1] + '_val_bkg_img'
    trn_bkg_img_dir = args.images_save_dir[:-1] + '_trn_bkg_img'
    cc_img_dir = args.images_save_dir[:-1] + '_cc{}'.format(cc_id)
    img_dir = args.images_save_dir

    all_ori_files = np.sort(glob.glob(os.path.join(all_ori_dir, '*.txt')))
    all_cc_files = np.sort(glob.glob(os.path.join(all_cc_dir, '*.txt')))
    all_cc_names = [os.path.basename(f) for f in all_cc_files]
    all_ncc_files = [f for f in all_ori_files if is_non_zero_file(f) and os.path.basename(f) not in all_cc_names]
    print('all_cc_files', len(all_cc_files))

    np.random.seed(seed)
    val_cc_num = round(len(all_cc_files)*args.val_percent)
    trn_cc_num = len(all_cc_files) - val_cc_num
    print('trn_cc_num, val_cc_num', trn_cc_num, val_cc_num)

    val_ncc_num = round(len(all_ncc_files)*args.val_percent)
    trn_ncc_num = len(all_ncc_files) - val_ncc_num
    print('trn_ncc_num, val_ncc_num', trn_ncc_num, val_ncc_num)

    all_cc_files = np.random.permutation(all_cc_files)
    trn_cc_files = all_cc_files[:trn_cc_num]
    val_cc_files = all_cc_files[trn_cc_num:trn_cc_num+val_cc_num]

    all_ncc_files = np.random.permutation(all_ncc_files)
    trn_ncc_files = all_ncc_files[:trn_ncc_num]
    val_ncc_files = all_ncc_files[trn_ncc_num:trn_ncc_num+val_ncc_num]
    trn_ncc_Lbl_dir = args.annos_save_dir[:-1] + '_ncc{}_trn_lbl'.format(cc_id)
    val_ncc_lbl_dir = args.annos_save_dir[:-1] + '_ncc{}_val_lbl'.format(cc_id)
    trn_ncc_img_dir = args.images_save_dir[:-1] + '_ncc{}_trn_img'.format(cc_id)
    val_ncc_img_dir = args.images_save_dir[:-1] + '_ncc{}_val_img'.format(cc_id)
    if not os.path.exists(trn_ncc_Lbl_dir):
        os.mkdir(trn_ncc_Lbl_dir)
    else:
        shutil.rmtree(trn_ncc_Lbl_dir)
        os.mkdir(trn_ncc_Lbl_dir)
    if not os.path.exists(val_ncc_lbl_dir):
        os.mkdir(val_ncc_lbl_dir)
    else:
        shutil.rmtree(val_ncc_lbl_dir)
        os.mkdir(val_ncc_lbl_dir)
    if not os.path.exists(trn_ncc_img_dir):
        os.mkdir(trn_ncc_img_dir)
    else:
        shutil.rmtree(trn_ncc_img_dir)
        os.mkdir(trn_ncc_img_dir)
    if not os.path.exists(val_ncc_img_dir):
        os.mkdir(val_ncc_img_dir)
    else:
        shutil.rmtree(val_ncc_img_dir)
        os.mkdir(val_ncc_img_dir)

    trn_bkg_lbl_files = np.sort(glob.glob(os.path.join(trn_bkg_lbl_dir, '*.txt')))
    val_bkg_lbl_files = np.sort(glob.glob(os.path.join(val_bkg_lbl_dir, '*.txt')))
    print('trn_bkg_lbl_files, val_bkg_lbl_files', len(trn_bkg_lbl_files), len(val_bkg_lbl_files))

    data_save_dir =  os.path.join(args.data_save_dir, base_pxwhrs, 'CC')
    if not os.path.exists(data_save_dir):
        os.mkdir(data_save_dir)

    trn_cc_img_txt = open(os.path.join(data_save_dir, 'only_cc{}_trn_img_{}.txt'.format(cc_id, base_pxwhrs)), 'w')
    trn_cc_lbl_txt = open(os.path.join(data_save_dir, 'only_cc{}_trn_lbl_{}.txt'.format(cc_id, base_pxwhrs)), 'w')
    trn_nccbkg_img_txt = open(os.path.join(data_save_dir, '{}_trn_img_{}.txt'.format(txt_name, base_pxwhrs)), 'w')
    trn_nccbkg_lbl_txt = open(os.path.join(data_save_dir, '{}_trn_lbl_{}.txt'.format(txt_name, base_pxwhrs)), 'w')

    cc_img_trn_dir = args.images_save_dir[:-1] + '_trn_cc{}'.format(cc_id)
    if not os.path.exists(cc_img_trn_dir):
        os.mkdir(cc_img_trn_dir)
    cc_lbl_trn_dir = args.annos_save_dir[:-1] + '_trn_cc{}_with_id'.format(cc_id)
    if not os.path.exists(cc_lbl_trn_dir):
        os.mkdir(cc_lbl_trn_dir)
    for f in trn_cc_files:
        trn_cc_lbl_txt.write("%s\n" % f)
        lbl_name = os.path.basename(f)
        img_name = lbl_name.replace('.txt', '.jpg')
        trn_cc_img_txt.write("%s\n" % os.path.join(cc_img_dir, img_name))
        shutil.copy(os.path.join(cc_img_dir, img_name), os.path.join(cc_img_trn_dir, img_name))
        shutil.copy(f, os.path.join(cc_lbl_trn_dir, lbl_name))

    for f in trn_ncc_files:
        trn_nccbkg_lbl_txt.write("%s\n" % f)
        lbl_name = os.path.basename(f)
        img_name = lbl_name.replace('.txt', '.jpg')
        trn_nccbkg_img_txt.write("%s\n" % os.path.join(img_dir, img_name))
        shutil.copy(os.path.join(img_dir, img_name), os.path.join(trn_ncc_img_dir, img_name))
        shutil.copy(f, os.path.join(trn_ncc_Lbl_dir, lbl_name))

    for f in trn_bkg_lbl_files:
        trn_nccbkg_lbl_txt.write("%s\n" % f)
        img_name = os.path.basename(f).replace('.txt', '.jpg')
        trn_nccbkg_img_txt.write("%s\n" % os.path.join(trn_bkg_img_dir, img_name))

    trn_nccbkg_lbl_txt.close()
    trn_nccbkg_img_txt.close()

    ###### validate xview_cc_nrc_bkg
    cc_img_val_dir = args.images_save_dir[:-1] + '_val_cc{}'.format(cc_id)
    if not os.path.exists(cc_img_val_dir):
        os.mkdir(cc_img_val_dir)
    cc_lbl_val_dir = args.annos_save_dir[:-1] + '_val_cc{}_with_id'.format(cc_id)
    if not os.path.exists(cc_lbl_val_dir):
        os.mkdir(cc_lbl_val_dir)

    val_nccbkg_cc_img_txt = open(os.path.join(data_save_dir, 'xview_nccbkg_cc{}_val_img_{}.txt'.format(cc_id, base_pxwhrs)), 'w')
    val_nccbkg_cc_lbl_txt = open(os.path.join(data_save_dir, 'xview_nccbkg_cc{}_val_lbl_{}.txt'.format(cc_id, base_pxwhrs)), 'w')

    for f in val_cc_files:
        val_nccbkg_cc_lbl_txt.write("%s\n" % f)
        lbl_name = os.path.basename(f)
        img_name = lbl_name.replace('.txt', '.jpg')
        val_nccbkg_cc_img_txt.write("%s\n" % os.path.join(cc_img_dir, img_name))
        shutil.copy(f, os.path.join(cc_lbl_val_dir, lbl_name))
        shutil.copy(os.path.join(cc_img_dir, img_name), os.path.join(cc_img_val_dir, img_name))

    for f in val_ncc_files:
        val_nccbkg_cc_lbl_txt.write("%s\n" % f)
        lbl_name = os.path.basename(f)
        img_name = lbl_name.replace('.txt', '.jpg')
        val_nccbkg_cc_img_txt.write("%s\n" % os.path.join(img_dir, img_name))
        shutil.copy(os.path.join(img_dir, img_name), os.path.join(val_ncc_img_dir, img_name))
        shutil.copy(f, os.path.join(val_ncc_lbl_dir, lbl_name))

    for f in val_bkg_lbl_files:
        val_nccbkg_cc_lbl_txt.write("%s\n" % f)
        img_name = os.path.basename(f).replace('.txt', '.jpg')
        val_nccbkg_cc_img_txt.write("%s\n" % os.path.join(val_bkg_img_dir, img_name))

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


def create_aug_cc_nccbkg_test(cc_id, seed=17):
    pxwhrs = 'px23whr3_seed17'
    eht = 'easy'
    base_dir = os.path.join(args.data_save_dir, pxwhrs, 'CC')
    test_lbl_txt = open(os.path.join(base_dir, 'xview_nccbkg_aug_cc{}_test_lbl_{}.txt'.format(cc_id, pxwhrs)), 'w')
    test_img_txt = open(os.path.join(base_dir, 'xview_nccbkg_aug_cc{}_test_img_{}.txt'.format(cc_id, pxwhrs)), 'w')
    lbl_dir = args.annos_save_dir[:-1] + '_val_cc{}_with_id_aug'.format(cc_id)
    img_dir = args.images_save_dir[:-1] + '_val_cc{}_aug'.format(cc_id)
    aug_cc_lbls = np.sort(glob.glob(os.path.join(lbl_dir, '*.txt')))
    aug_cc_lbl_names = [os.path.basename(f) for f in aug_cc_lbls]
    for f in aug_cc_lbls:
        test_lbl_txt.write('%s\n' % f)
        name = os.path.basename(f).replace('.txt', '.jpg')
        test_img_txt.write('%s\n' % os.path.join(img_dir, name))

    val_ncc_lbl_dir = args.annos_save_dir[:-1] + '_ncc{}_val_lbl'.format(cc_id)
    val_ncc_img_dir = args.images_save_dir[:-1] + '_ncc{}_val_img'.format(cc_id)
    val_ncc_lbl_files = np.sort(glob.glob(os.path.join(val_ncc_lbl_dir, '*.txt')))
    print('val_ncc_lbl_files', len(val_ncc_lbl_files), val_ncc_img_dir)
    for f in val_ncc_lbl_files:
        test_lbl_txt.write('%s\n' % f)
        name = os.path.basename(f).replace('.txt', '.jpg')
        test_img_txt.write('%s\n' % os.path.join(val_ncc_img_dir, name))

    val_bkg_lbl_dir = args.annos_save_dir[:-1] + '_val_bkg_lbl'
    val_bkg_img_dir = args.images_save_dir[:-1] + '_val_bkg_img'
    val_bkg_lbl_files = np.sort(glob.glob(os.path.join(val_bkg_lbl_dir, '*.txt')))
    for f in val_bkg_lbl_files:
        test_lbl_txt.write('%s\n' % f)
        name = os.path.basename(f).replace('.txt', '.jpg')
        test_img_txt.write('%s\n' % os.path.join(val_bkg_img_dir, name))

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
    parser.add_argument("--data_list_save_dir", type=str, help="to save data files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_cls/data_list/')
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
    args.data_list_save_dir = args.data_list_save_dir.format(args.input_size, args.class_num)
    if not os.path.exists(args.txt_save_dir):
        os.makedirs(args.txt_save_dir)

    if not os.path.exists(args.annos_save_dir):
        os.makedirs(args.annos_save_dir)
    if not os.path.exists(args.data_list_save_dir):
        os.makedirs(args.data_list_save_dir)
    if not os.path.exists(args.images_save_dir):
        os.makedirs(args.images_save_dir)

    if not os.path.exists(args.data_save_dir):
        os.makedirs(args.data_save_dir)
    if not os.path.exists(args.data_list_save_dir):
        os.makedirs(args.data_list_save_dir)
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
    remove ept files (drop by b-box rules: px_thres<23, whr_thres>3)
    from lbl files that contain airplanes
    '''
    # move_ept_to_bkg()
    '''
    split BKG images into train and val
    '''
    # comments = 'px{}whr{}_seed{}'
    # px_thres = 23
    # whr_thres = 3
    # seed = 17
    # split_bkg_into_train_val(comments, seed, px_thres, whr_thres)
    '''
    split CC1, CC2 separately
    1. split CC* to train and val
    2. split Non-CC to train and val
    3. combine BKG of train and val
    '''
    seed = 17
    base_pxwhrs = 'px{}whr{}_seed{}'.format(px_thres, whr_thres, seed)
    # cc_id = 1
    cc_id = 2
    txt_name = 'xview_nccbkg_nocc{}'.format(cc_id)
    split_trn_val_for_each_CC_step_by_step(cc_id, seed, txt_name, base_pxwhrs)
    data_name = 'xview_nccbkg_cc{}'.format(cc_id)
    create_xview_cc_nccbkg_data(cc_id, data_name, txt_name, seed)

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
    create test data of Aug CC
    '''
    # cc_id = 1
    cc_id = 2
    create_aug_cc_nccbkg_test(cc_id)

    '''
    create *.data with val augmented CC
    '''
    seed = 17
    # cc_id = 1
    cc_id = 2
    txt_name = 'xview_nccbkg_nocc{}'.format(cc_id)
    data_name = 'xview_nccbkg_aug_cc{}'.format(cc_id)
    create_xview_cc_nccbkg_data(cc_id, data_name, txt_name, seed, val_aug=True)


