import os
import argparse
import glob
import shutil
import numpy as np
import pandas as pd
from utils.object_score_util import get_bbox_coords_from_annos_with_object_score as gbc

def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def label_cc_and_rc_remove_other_airp_labels():
    '''
    remove other airplane labels ccid=None, rcid=None
    :return:
    '''
    all_rc_lbl_dir = args.annos_save_dir[:-1] + '_rc_all_ori_rcid'
    all_rc_lbl0_dir = args.annos_save_dir[:-1] + '_rc_all_ori_rcid0'
    if not os.path.exists(all_rc_lbl0_dir):
        os.mkdir(all_rc_lbl0_dir)
    all_rc_lbl_files = glob.glob(os.path.join(all_rc_lbl_dir, '*.txt'))
    for f in all_rc_lbl_files:
        df = pd.read_csv(f, header=None, sep=' ')
        df = df.drop(df[df.loc[:, 5] == 0].index)
        df.to_csv(f, header=False, index=False, sep=' ')
        df.loc[:, 5] = 0
        lbl_name = os.path.basename(f)
        df.to_csv(os.path.join(all_rc_lbl0_dir, lbl_name), header=False, index=False, sep=' ')

    cc1_lbl_dir = args.annos_save_dir[:-1] + '_cc1_with_id'
    cc1_lbl0_dir = args.annos_save_dir[:-1] + '_cc1_with_id0'
    if not os.path.exists(cc1_lbl0_dir):
        os.mkdir(cc1_lbl0_dir)
    all_cc1_lbl_files = glob.glob(os.path.join(cc1_lbl_dir, '*.txt'))
    for f in all_cc1_lbl_files:
        df = pd.read_csv(f, header=None, sep=' ')
        df = df.drop(df[df.loc[:, 5] == 0].index)
        df.to_csv(f, header=False, index=False, sep=' ')
        df.loc[:, 5] = 0
        lbl_name = os.path.basename(f)
        df.to_csv(os.path.join(cc1_lbl0_dir, lbl_name), header=False, index=False, sep=' ')

    cc2_lbl_dir = args.annos_save_dir[:-1] + '_cc2_with_id'
    cc2_lbl0_dir = args.annos_save_dir[:-1] + '_cc2_with_id0'
    if not os.path.exists(cc2_lbl0_dir):
        os.mkdir(cc2_lbl0_dir)
    all_cc2_lbl_files = glob.glob(os.path.join(cc2_lbl_dir, '*.txt'))
    for f in all_cc2_lbl_files:
        # print('f', f)
        df = pd.read_csv(f, header=None, sep=' ')
        df = df.drop(df[df.loc[:, 5] == 0].index)
        df.to_csv(f, header=False, index=False, sep=' ')
        df.loc[:, 5] = 0
        lbl_name = os.path.basename(f)
        df.to_csv(os.path.join(cc2_lbl0_dir, lbl_name), header=False, index=False, sep=' ')


def plot_bbox_on_images_of_rc_and_cc():
    img_dir = args.images_save_dir
    bbox_folder_name = 'all_ori_rc_images_with_bbox_with_rcid_hard'
    rc_save_dir = os.path.join(args.cat_sample_dir, bbox_folder_name)
    if not os.path.exists(rc_save_dir):
        os.mkdir(rc_save_dir)
    all_rc_lbl_dir = args.annos_save_dir[:-1] + '_rc_all_ori_rcid'
    all_rc_lbl_files = glob.glob(os.path.join(all_rc_lbl_dir, '*.txt'))
    for f in all_rc_lbl_files:
        img_name = os.path.basename(f).replace('.txt', '.jpg')
        img_file = os.path.join(img_dir, img_name)
        gbc.plot_img_with_bbx(img_file, f, rc_save_dir,  rare_id=True)

    ccids = [1, 2]
    for cc_id in ccids:
        bbox_folder_name = 'cc{}_images_with_bbox_with_ccid_hard'.format(cc_id)
        cc_save_dir = os.path.join(args.cat_sample_dir, bbox_folder_name)
        if not os.path.exists(cc_save_dir):
            os.mkdir(cc_save_dir)
        cc_lbl_dir = args.annos_save_dir[:-1] + '_cc{}_with_id'.format(cc_id)
        cc_lbl_files = glob.glob(os.path.join(cc_lbl_dir, '*.txt'))
        for f in cc_lbl_files:
            img_name = os.path.basename(f).replace('.txt', '.jpg')
            img_file = os.path.join(img_dir, img_name)
            gbc.plot_img_with_bbx(img_file, f, cc_save_dir,  rare_id=True)


def split_cc_ncc_and_rc_by_cc_id(ccid, seed=17):
    '''
    Note RC2 is a subset of CC2
    split CC* to train:val=80%:20%
    :param ccid:
    :return:
    '''
    all_rc_lbl_dir = args.annos_save_dir[:-1] + '_rc_all_ori_rcid'
    all_rc_files = glob.glob(os.path.join(all_rc_lbl_dir, '*.txt'))
    all_rc_names = [os.path.basename(f) for f in all_rc_files]
    all_rc_lbl0_dir = args.annos_save_dir[:-1] + '_rc_all_ori_rcid0'
    if not os.path.exists(all_rc_lbl0_dir):
        os.mkdir(all_rc_lbl0_dir)
    for f in all_rc_files:
        df = pd.read_csv(f, header=None, sep=' ')
        df.loc[:, 5] = 0
        lbl_name = os.path.basename(f)
        df.to_csv(os.path.join(all_rc_lbl0_dir, lbl_name), header=False, index=False, sep=' ')

    cc_lbl_dir = args.annos_save_dir[:-1] + '_cc{}_with_id'.format(ccid)
    cc_lbl0_dir = args.annos_save_dir[:-1] + '_cc{}_with_id0'.format(ccid)
    if not os.path.exists(cc_lbl0_dir):
        os.mkdir(cc_lbl0_dir)
    cc_lbl_files = np.sort(glob.glob(os.path.join(cc_lbl_dir, '*.txt')))
    for f in cc_lbl_files:
        df = pd.read_csv(f, header=None, sep=' ')
        df.loc[:, 5] = 0
        lbl_name = os.path.basename(f)
        df.to_csv(os.path.join(cc_lbl0_dir, lbl_name), header=False, index=False, sep=' ')

    np.random.seed(seed)
    val_cc_num = int(len(cc_lbl_files)*args.val_percent)
    trn_cc_num = len(cc_lbl_files) - val_cc_num
    #split CC into val and trn
    ##### if subset of rc is not None, then subset of rc belongs to val
    ## make sure that patches of subset belong to val (unique)
    rc_subset = [f for f in cc_lbl_files if os.path.basename(f) in all_rc_names]
    print('rc_subset', len(rc_subset))
    print('rc_subset', rc_subset)
    new_cc_lbl_files = [f for f in cc_lbl_files if f not in rc_subset]
    new_cc_lbl_files = np.random.permutation(new_cc_lbl_files)
    if len(rc_subset):
        val_cc_num = val_cc_num - len(rc_subset)
        val_cc_lbl_files = new_cc_lbl_files[:val_cc_num].tolist() + rc_subset
    else:
        val_cc_lbl_files = new_cc_lbl_files[:val_cc_num]
    print('val_cc_lbl_files', len(val_cc_lbl_files))
    trn_cc_lbl_files = new_cc_lbl_files[val_cc_num : val_cc_num + trn_cc_num]
    print('trn_cc_lbl_files', len(trn_cc_lbl_files))

    trn_cc_lbl_dir = args.annos_save_dir[:-1] + '_cc{}_with_id_trn'.format(ccid)
    if not os.path.exists(trn_cc_lbl_dir):
        os.mkdir(trn_cc_lbl_dir)
    val_cc_lbl_dir = args.annos_save_dir[:-1] + '_cc{}_with_id_val'.format(ccid)
    if not os.path.exists(val_cc_lbl_dir):
        os.mkdir(val_cc_lbl_dir)
    images_save_dir = args.images_save_dir
    trn_cc_img_dir = args.images_save_dir[:-1] + '_cc{}_trn'.format(ccid)
    if not os.path.exists(trn_cc_img_dir):
        os.mkdir(trn_cc_img_dir)
    val_cc_img_dir = args.images_save_dir[:-1] + '_cc{}_val'.format(ccid)
    if not os.path.exists(val_cc_img_dir):
        os.mkdir(val_cc_img_dir)

    for f in val_cc_lbl_files:
        lbl_name = os.path.basename(f)
        img_name = lbl_name.replace('.txt', '.jpg')
        shutil.copy(f, os.path.join(val_cc_lbl_dir, lbl_name))
        shutil.copy(os.path.join(images_save_dir, img_name), os.path.join(val_cc_img_dir, img_name))
    for f in trn_cc_lbl_files:
        lbl_name = os.path.basename(f)
        img_name = lbl_name.replace('.txt', '.jpg')
        shutil.copy(f, os.path.join(trn_cc_lbl_dir, lbl_name))
        shutil.copy(os.path.join(images_save_dir, img_name), os.path.join(trn_cc_img_dir, img_name))


def add_ncc_nrc_to_bkg():
    '''
    Other airplane(NCC, NRC) labels belong to BG
    '''
    bkg_lbl_dir = args.annos_save_dir[:-1] + '_bkg'
    bkg_img_dir = args.images_save_dir[:-1] + '_noairplane_bkg_chips'
    cc1_lbl0_dir = args.annos_save_dir[:-1] + '_cc1_with_id0'
    cc2_lbl0_dir = args.annos_save_dir[:-1] + '_cc2_with_id0'
    all_rc_lbl_dir = args.annos_save_dir[:-1] + '_rc_all_ori_rcid'
    all_airp_lbl_dir = args.annos_save_dir[:-1] + '_with_id0'
    all_airp_img_dir = args.images_save_dir
    cc1_lbl_files = glob.glob(os.path.join(cc1_lbl0_dir, '*.txt'))
    cc2_lbl_files = glob.glob(os.path.join(cc2_lbl0_dir, '*.txt'))
    rc_lbl_files = glob.glob(os.path.join(all_rc_lbl_dir, '*.txt'))
    all_lbl_files = glob.glob(os.path.join(all_airp_lbl_dir, '*.txt'))
    all_cc_rc_files = cc1_lbl_files + cc2_lbl_files + rc_lbl_files
    all_cc_rc_names = [os.path.basename(f) for f in all_cc_rc_files]
    cnt = 0
    for f in all_lbl_files:
        lbl_name = os.path.basename(f)
        if lbl_name not in all_cc_rc_names:
            ###### remove labesl of other airplanes
            txt_file = open(os.path.join(bkg_lbl_dir, lbl_name), 'w')
            txt_file.close()
            img_name = lbl_name.replace('.txt', '.jpg')
            shutil.copy(os.path.join(all_airp_img_dir, img_name), os.path.join(bkg_img_dir, img_name))
            cnt += 1
    print('cnt', cnt)


def split_bkg_into_train_val(comments='px{}whr{}_seed{}', seed=17):
    '''
    Other airplane(NCC, NRC) labels belong to BG
    split data contains no aircrafts (bkg images)
    '''
    comments = comments.format(px_thres, whr_thres, seed)
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


def combine_trn_val_for_each_CC_step_by_step(cc_id, base_pxwhrs, ccids=[1, 2]):
    '''
    split CC1, CC2 separately
    train: CC(i) + Non-CC(i) + BKG
    val: CC(i) + Non-CC(i) + BKG + RC
    :return:
    '''

    ori_rc_lbl_dir = args.annos_save_dir[:-1] + '_rc_all_ori_rcid'
    all_aug_rc_lbl_dir = ori_rc_lbl_dir + '_flip_aug'
    all_aug_rc_img_dir = args.images_save_dir[:-1] + '_rc_ori_flip_aug'

    trn_cc_lbl_dir =  args.annos_save_dir[:-1] + '_cc{}_with_id_trn'.format(cc_id)
    trn_cc_img_dir =  args.images_save_dir[:-1] + '_cc{}_trn'.format(cc_id)
    val_cc_lbl_dir =  args.annos_save_dir[:-1] + '_cc{}_with_id_val'.format(cc_id)
    val_cc_img_dir =  args.images_save_dir[:-1] + '_cc{}_val'.format(cc_id)

    non_ccid = ccids[ccids != cc_id][0]
    print('non_ccid', non_ccid)
    trn_ncc_lbl_dir = args.annos_save_dir[:-1] + '_cc{}_with_id_trn'.format(non_ccid)
    val_ncc_lbl_dir = args.annos_save_dir[:-1] + '_cc{}_with_id_val'.format(non_ccid)
    # ncc_lbl0_dir =  args.annos_save_dir[:-1] + '_cc{}_with_id0'.format(non_ccid)
    trn_ncc_img_dir = args.images_save_dir[:-1] + '_cc{}_trn'.format(non_ccid)
    val_ncc_img_dir = args.images_save_dir[:-1] + '_cc{}_val'.format(non_ccid)

    val_bkg_lbl_dir = args.annos_save_dir[:-1] + '_val_bkg_lbl'
    trn_bkg_lbl_dir = args.annos_save_dir[:-1] + '_trn_bkg_lbl'
    val_bkg_img_dir = args.images_save_dir[:-1] + '_val_bkg_img'
    trn_bkg_img_dir = args.images_save_dir[:-1] + '_trn_bkg_img'

    cc_img_dir = args.images_save_dir[:-1] + '_cc{}'.format(cc_id)
    img_dir = args.images_save_dir

    all_ori_rc_files = np.sort(glob.glob(os.path.join(ori_rc_lbl_dir, '*.txt')))
    all_ori_rc_names = [os.path.basename(f) for f in all_ori_rc_files]
    all_aug_rc_lbl_files = np.sort(glob.glob(os.path.join(all_aug_rc_lbl_dir, '*.txt')))

    # all_ncc_files = [f for f in all_ori_files if is_non_zero_file(f) and os.path.basename(f) not in all_cc_names]
    # print('all_cc_files', len(all_cc_files))

    trn_cc_lbl_files = np.sort(glob.glob(os.path.join(trn_cc_lbl_dir, '*.txt')))
    val_cc_lbl_files = np.sort(glob.glob(os.path.join(val_cc_lbl_dir, '*.txt')))
    # cc_lbl0_files = np.sort(glob.glob(os.path.join(cc_lbl0_dir, '*.txt')))
    print('trn_cc_lbl_files, val_cc_lbl_files', len(trn_cc_lbl_files), len(val_cc_lbl_files))

    trn_ncc_img_files = np.sort(glob.glob(os.path.join(trn_ncc_img_dir, '*.jpg')))
    val_ncc_img_files = np.sort(glob.glob(os.path.join(val_ncc_img_dir, '*.jpg')))
    print('trn_ncc_img_files, val_ncc_img_files', len(trn_ncc_img_files), len(val_ncc_img_files))

    trn_bkg_lbl_files = np.sort(glob.glob(os.path.join(trn_bkg_lbl_dir, '*.txt')))
    val_bkg_lbl_files = np.sort(glob.glob(os.path.join(val_bkg_lbl_dir, '*.txt')))
    print('trn_bkg_lbl_files, val_bkg_lbl_files', len(trn_bkg_lbl_files), len(val_bkg_lbl_files))

    data_save_dir =  os.path.join(args.data_save_dir, base_pxwhrs, 'CC')
    if not os.path.exists(data_save_dir):
        os.mkdir(data_save_dir)
    trn_cc_img_txt = open(os.path.join(data_save_dir, 'only_cc{}_trn_img_{}.txt'.format(cc_id, base_pxwhrs)), 'w')
    trn_cc_lbl_txt = open(os.path.join(data_save_dir, 'only_cc{}_trn_lbl_{}.txt'.format(cc_id, base_pxwhrs)), 'w')
    trn_nccbkg_img_txt = open(os.path.join(data_save_dir, 'xview_ncc{}bkg_trn_img_{}.txt'.format(cc_id, base_pxwhrs)), 'w')
    trn_nccbkg_lbl_txt = open(os.path.join(data_save_dir, 'xview_ncc{}bkg_trn_lbl_{}.txt'.format(cc_id, base_pxwhrs)), 'w')
    trn_nccbkg_cc_img_txt = open(os.path.join(data_save_dir, 'xview_oa_bkg_cc{}_trn_img_{}.txt'.format(cc_id, base_pxwhrs)), 'w')
    trn_nccbkg_cc_lbl_txt = open(os.path.join(data_save_dir, 'xview_oa_bkg_cc{}_trn_lbl_{}.txt'.format(cc_id, base_pxwhrs)), 'w')

    for f in trn_cc_lbl_files:
        trn_cc_lbl_txt.write("%s\n" % f)
        lbl_name = os.path.basename(f)
        img_name = lbl_name.replace('.txt', '.jpg')
        trn_cc_img_txt.write("%s\n" % os.path.join(trn_cc_img_dir, img_name))
        trn_nccbkg_cc_lbl_txt.write("%s\n" % f)
        trn_nccbkg_cc_img_txt.write("%s\n" % os.path.join(trn_cc_img_dir, img_name))
    trn_cc_lbl_txt.close()
    trn_cc_img_txt.close()

    for f in trn_ncc_img_files:
        trn_nccbkg_img_txt.write("%s\n" % f)
        img_name = os.path.basename(f)
        lbl_name = img_name.replace('.jpg', '.txt')
        trn_nccbkg_lbl_txt.write("%s\n" % os.path.join(trn_ncc_lbl_dir, lbl_name))
        trn_nccbkg_cc_img_txt.write("%s\n" % f)
        trn_nccbkg_cc_lbl_txt.write("%s\n" % os.path.join(trn_ncc_lbl_dir, lbl_name))


    for f in trn_bkg_lbl_files:
        trn_nccbkg_lbl_txt.write("%s\n" % f)
        img_name = os.path.basename(f).replace('.txt', '.jpg')
        trn_nccbkg_img_txt.write("%s\n" % os.path.join(trn_bkg_img_dir, img_name))
        trn_nccbkg_cc_lbl_txt.write("%s\n" % f)
        trn_nccbkg_cc_img_txt.write("%s\n" % os.path.join(trn_bkg_img_dir, img_name))

    trn_nccbkg_lbl_txt.close()
    trn_nccbkg_img_txt.close()
    trn_nccbkg_cc_lbl_txt.close()
    trn_nccbkg_cc_img_txt.close()

    ###### validate xview_cc_nrc_bkg
    val_cc_img_txt = open(os.path.join(data_save_dir, 'only_cc{}_val_img_{}.txt'.format(cc_id, base_pxwhrs)), 'w')
    val_cc_lbl_txt = open(os.path.join(data_save_dir, 'only_cc{}_val_lbl_{}.txt'.format(cc_id, base_pxwhrs)), 'w')
    val_nccbkg_cc_img_txt = open(os.path.join(data_save_dir, 'xview_oa_bkg_cc{}_val_img_{}.txt'.format(cc_id, base_pxwhrs)), 'w')
    val_nccbkg_cc_lbl_txt = open(os.path.join(data_save_dir, 'xview_oa_bkg_cc{}_val_lbl_{}.txt'.format(cc_id, base_pxwhrs)), 'w')

    rc_subset = []
    for f in val_cc_lbl_files:
        val_cc_lbl_txt.write("%s\n" % f)
        val_nccbkg_cc_lbl_txt.write("%s\n" % f)
        lbl_name = os.path.basename(f)
        if lbl_name in all_ori_rc_names:
            rc_subset.append(lbl_name)
        img_name = lbl_name.replace('.txt', '.jpg')
        val_cc_img_txt.write("%s\n" % os.path.join(val_cc_img_dir, img_name))
        val_nccbkg_cc_img_txt.write("%s\n" % os.path.join(val_cc_img_dir, img_name))
    val_cc_lbl_txt.close()
    val_cc_img_txt.close()

    ori_subset = len(rc_subset)
    for f in val_ncc_img_files:
        img_name = os.path.basename(f)
        lbl_name = img_name.replace('.jpg', '.txt')
        # if rc_subset is empty
        if not ori_subset and lbl_name in all_ori_rc_names:
            rc_subset.append(lbl_name)
        val_nccbkg_cc_img_txt.write("%s\n" % f)
        val_nccbkg_cc_lbl_txt.write("%s\n" % os.path.join(val_ncc_lbl_dir, lbl_name))

    cnt = 0
    for f in all_aug_rc_lbl_files:
        lbl_name = os.path.basename(f)
        # drop dumplicate
        if lbl_name in rc_subset:
            # print('rc subset', lbl_name)
            cnt += 1
            continue
        val_nccbkg_cc_lbl_txt.write("%s\n" % f)
        img_name = lbl_name.replace('.txt', '.jpg')
        val_nccbkg_cc_img_txt.write("%s\n" % os.path.join(all_aug_rc_img_dir, img_name))

    print('all_aug_rc_files', len(all_aug_rc_lbl_files), 'cnt of rc', cnt)

    for f in val_bkg_lbl_files:
        val_nccbkg_cc_lbl_txt.write("%s\n" % f)
        img_name = os.path.basename(f).replace('.txt', '.jpg')
        val_nccbkg_cc_img_txt.write("%s\n" % os.path.join(val_bkg_img_dir, img_name))

    val_nccbkg_cc_lbl_txt.close()
    val_nccbkg_cc_img_txt.close()


def create_xview_cc_nccbkg_data(cc_id, data_name, seed=17):
    base_cmt = 'px{}whr{}_seed{}'.format(px_thres, whr_thres, seed)
    data_save_dir =  os.path.join(args.data_save_dir, base_cmt, 'CC')
    print('data_save_dir', data_save_dir)
    data_txt = open(os.path.join(data_save_dir, '{}_{}.data'.format(data_name, base_cmt)), 'w')
    data_txt.write(
        'xview_train={}\n'.format(os.path.join(data_save_dir, 'xview_ncc{}bkg_trn_img_{}.txt'.format(cc_id, base_cmt))))
    data_txt.write(
        'xview_train_label={}\n'.format(os.path.join(data_save_dir, 'xview_ncc{}bkg_trn_lbl_{}.txt'.format(cc_id, base_cmt))))

    data_txt.write(
        'cc_train={}\n'.format(os.path.join(data_save_dir,  'only_cc{}_trn_img_{}.txt'.format(cc_id, base_cmt))))
    data_txt.write(
        'cc_train_label={}\n'.format(os.path.join(data_save_dir, 'only_cc{}_trn_lbl_{}.txt'.format(cc_id, base_cmt))))

    data_txt.write(
        'valid={}\n'.format(os.path.join(data_save_dir, 'xview_oa_bkg_cc{}_val_img_{}.txt'.format(cc_id, base_cmt))))
    data_txt.write(
        'valid_label={}\n'.format(os.path.join(data_save_dir, 'xview_oa_bkg_cc{}_val_lbl_{}.txt'.format(cc_id, base_cmt))))

    df_trn_nccbkg = pd.read_csv(os.path.join(data_save_dir, 'xview_ncc{}bkg_trn_img_{}.txt'.format(cc_id, base_cmt)), header=None)
    df_trn_cc = pd.read_csv(os.path.join(data_save_dir,  'only_cc{}_trn_img_{}.txt'.format(cc_id, base_cmt)), header=None)
    xview_trn_num = df_trn_nccbkg.shape[0] + df_trn_cc.shape[0]

    data_txt.write('xview_number={}\n'.format(xview_trn_num))
    data_txt.write('classes=%s\n' % str(args.class_num))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval=color\n')
    data_txt.close()


def combine_syn_xview_cc(comment, seed=17, base_cmt='px23whr3_seed17', bkg_cc_sep=False):
    cc_id = comment.find('CC')
    data_xview_dir =  os.path.join(args.data_xview_dir, base_cmt, 'CC')
    print('data_xview_dir', data_xview_dir)
    xview_img_txt = pd.read_csv(open(os.path.join(data_xview_dir, '{}_train_img_{}.txt'.format(name, base_cmt))), header=None).to_numpy()
    xview_trn_num = xview_img_txt.shape[0]
    data_txt = open(os.path.join(data_xview_dir, 'syn_xview_cc{}_{}_seed{}.data'.format(cc_id, comment, seed)), 'w')
    if not bkg_cc_sep: # cc and bkg are not separated
        data_txt.write(
            'xview_train={}\n'.format(os.path.join(data_xview_dir, 'xview_oa_bkg_cc{}_trn_img_{}.txt'.format(cc_id, base_cmt))))
        data_txt.write(
            'xview_train_label={}\n'.format(os.path.join(data_xview_dir, 'xview_oa_bkg_cc{}_trn_lbl_{}.txt'.format(cc_id, base_cmt))))
    else: # mixed batch of cc and bkg
        data_txt.write(
            'xview_cc_train={}\n'.format(os.path.join(data_xview_dir, 'only_cc{}_trn_img_{}.txt'.format(cc_id, base_cmt))))
        data_txt.write(
            'xview_cc_train_label={}\n'.format(os.path.join(data_xview_dir, 'only_cc{}_trn_lbl_{}.txt'.format(cc_id, base_cmt))))
        data_txt.write(
            'xview_bkg_train={}\n'.format(os.path.join(data_xview_dir, 'xview_ncc{}bkg_trn_img_{}.txt'.format(cc_id, base_cmt))))
        data_txt.write(
            'xview_bkg_train_label={}\n'.format(os.path.join(data_xview_dir, 'xview_ncc{}bkg_trn_lbl_{}.txt'.format(cc_id, base_cmt))))

    syn_data_dir = args.syn_data_dir.format(comment, args.class_num)
    data_txt.write(
        'syn_train={}\n'.format(os.path.join(syn_data_dir, '{}_seed{}'.format(comment, seed),  '{}_train_img_seed{}.txt'.format(comment, seed))))
    data_txt.write(
        'syn_train_label={}\n'.format(os.path.join(syn_data_dir, '{}_seed{}'.format(comment, seed), '{}_train_lbl_seed{}.txt'.format(comment, seed))))

    data_txt.write(
        'valid={}\n'.format(os.path.join(data_xview_dir, 'xview_oa_bkg_cc{}_val_img_{}.txt'.format(cc_id, base_cmt))))
    data_txt.write(
        'valid_label={}\n'.format(os.path.join(data_xview_dir, 'xview_oa_bkg_cc{}_val_lbl_{}.txt'.format(cc_id, base_cmt))))

    data_txt.write('syn_0_xview_number={}\n'.format(xview_trn_num))
    data_txt.write('classes=%s\n' % str(args.class_num))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval=color\n')
    data_txt.close()



def augment_all_RC():
    '''
    augment RC(i)
    1. left-right flip 2x
    2. based on flipped patches, rotate 90, 180, 270 8x
    :return:
    '''
    from utils.xview_synthetic_util.resize_flip_rotate_images import flip_images, rotate_images, flip_rotate_coordinates
    ori_rc_img_dir = os.path.join(args.images_save_dir[:-1] + '_rc_ori')
    ori_rc_lbl_dir = args.annos_save_dir[:-1] + '_rc_all_ori_rcid'
    ori_rc_img_files = np.sort(glob.glob(os.path.join(ori_rc_img_dir, '*.jpg')))
    flipped_rc_img_dir = ori_rc_img_dir + '_flip'
    if not os.path.exists(flipped_rc_img_dir):
        os.mkdir(flipped_rc_img_dir)

    flipped_rc_lbl_dir = ori_rc_lbl_dir + '_flip'
    if not os.path.exists(flipped_rc_lbl_dir):
        os.mkdir(flipped_rc_lbl_dir)
    save_dir = args.cat_sample_dir + 'image_with_bbox/RC_flip/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for f in ori_rc_img_files:
        img_name = os.path.basename(f)
        img_dir = os.path.dirname(f)
        flip_images(img_name, img_dir)
        name = img_name.split('.')[0]
        flip_rotate_coordinates(flipped_rc_img_dir,ori_rc_lbl_dir, save_dir, flipped_rc_lbl_dir, name, flip='lr')

    flipped_img_files = np.sort(glob.glob(os.path.join(flipped_rc_img_dir, '*.jpg')))
    aug_img_dir = flipped_rc_img_dir + '_aug'
    if not os.path.exists(aug_img_dir):
        os.mkdir(aug_img_dir)
    aug_lbl_dir = flipped_rc_lbl_dir + '_aug'
    if not os.path.exists(aug_lbl_dir):
        os.mkdir(aug_lbl_dir)
    save_dir = args.cat_sample_dir + 'image_with_bbox/RC_aug/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for f in flipped_img_files:
        img_name = os.path.basename(f)
        img_dir = os.path.dirname(f)
        rotate_images(img_name, img_dir)
        name = img_name.split('.')[0]

        angle_list = [90, 180, 270]
        for angle in angle_list:
            flip_rotate_coordinates(aug_img_dir, flipped_rc_lbl_dir, save_dir, aug_lbl_dir, name, angle=angle)



def augment_RC_by_id(rc_id):
    '''
    augment RC(i)
    1. left-right flip 2x
    2. based on flipped patches, rotate 90, 180, 270 8x
    :return:
    '''
    from utils.xview_synthetic_util.resize_flip_rotate_images import flip_images, rotate_images, flip_rotate_coordinates
    ori_rc_img_dir = os.path.join(args.images_save_dir[:-1] + '_rc', 'rc{}'.format(rc_id))
    ori_rc_img_files = np.sort(glob.glob(os.path.join(ori_rc_img_dir, '*.jpg')))
    flipped_rc_img_dir = ori_rc_img_dir + '_flip'
    if not os.path.exists(flipped_rc_img_dir):
        os.mkdir(flipped_rc_img_dir)
    flipped_rc_lbl_dir = os.path.join(args.annos_save_dir[:-1] + '_rc', 'rc{}_flip'.format(rc_id))
    if not os.path.exists(flipped_rc_lbl_dir):
        os.makedirs(flipped_rc_lbl_dir)
    save_dir = args.cat_sample_dir + 'image_with_bbox/RC_flip/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    all_ori_rc_lbl_dir = args.annos_save_dir[:-1] + '_rc_all_ori_rcid'
    for f in ori_rc_img_files:
        img_name = os.path.basename(f)
        img_dir = os.path.dirname(f)
        flip_images(img_name, img_dir)
        name = img_name.split('.')[0]
        flip_rotate_coordinates(flipped_rc_img_dir, all_ori_rc_lbl_dir, save_dir, flipped_rc_lbl_dir, name, flip='lr')

    flipped_img_files = np.sort(glob.glob(os.path.join(flipped_rc_img_dir, '*.jpg')))
    aug_img_dir = flipped_rc_img_dir + '_aug'
    if not os.path.exists(aug_img_dir):
        os.mkdir(aug_img_dir)
    aug_lbl_dir = flipped_rc_lbl_dir + '_aug'
    if not os.path.exists(aug_lbl_dir):
        os.mkdir(aug_lbl_dir)
    save_dir = args.cat_sample_dir + 'image_with_bbox/RC_aug/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for f in flipped_img_files:
        img_name = os.path.basename(f)
        img_dir = os.path.dirname(f)
        rotate_images(img_name, img_dir)
        name = img_name.split('.')[0]

        angle_list = [90, 180, 270]
        for angle in angle_list:
            flip_rotate_coordinates(aug_img_dir, flipped_rc_lbl_dir, save_dir, aug_lbl_dir, name, angle=angle)


def combine_val_for_each_aug_RC_step_by_step(rc_id, base_pxwhrs, rcids):
    '''
    RC validation set
    val: AUG RC(i) + AUG Non-RC(i) + CC + BKG
    :return:
    '''
    ori_rc_img_dir = os.path.join(args.images_save_dir[:-1] + '_rc', 'rc{}'.format(rc_id))
    all_ori_rc_lbl_dir = args.annos_save_dir[:-1] + '_rc_all_ori_rcid'
    all_ori_rc_lbl_files = np.sort(glob.glob(os.path.join(all_ori_rc_lbl_dir, '*.txt')))
    all_ori_lbl_names = [os.path.basename(f) for f in all_ori_rc_lbl_files]

    aug_rc_lbl_dir = os.path.join(args.annos_save_dir[:-1] + '_rc', 'rc{}_flip_aug'.format(rc_id))
    aug_rc_img_dir = os.path.join(args.images_save_dir[:-1] + '_rc', 'rc{}_flip_aug'.format(rc_id))
    aug_rc_img_files = np.sort(glob.glob(os.path.join(aug_rc_img_dir, '*.jpg')))
    print('aug_rc{}_img_files'.format(rc_id), len(aug_rc_img_files))

    data_save_dir =  os.path.join(args.data_save_dir, base_pxwhrs, 'RC')
    if not os.path.exists(data_save_dir):
        os.mkdir(data_save_dir)
    val_rc_img_txt = open(os.path.join(data_save_dir, 'only_aug_rc{}_val_img_{}.txt'.format(rc_id, base_pxwhrs)), 'w')
    val_rc_lbl_txt = open(os.path.join(data_save_dir, 'only_aug_rc{}_val_lbl_{}.txt'.format(rc_id, base_pxwhrs)), 'w')
    val_rc_nrcbkg_img_txt = open(os.path.join(data_save_dir, 'xview_oa_bkg_aug_rc{}_val_img_{}.txt'.format(rc_id, base_pxwhrs)), 'w')
    val_rc_nrcbkg_lbl_txt = open(os.path.join(data_save_dir, 'xview_oa_bkg_aug_rc{}_val_lbl_{}.txt'.format(rc_id, base_pxwhrs)), 'w')

    for f in aug_rc_img_files:
        val_rc_img_txt.write("%s\n" % f)
        img_name = os.path.basename(f)
        lbl_name = img_name.replace('.jpg', '.txt')
        val_rc_lbl_txt.write("%s\n" % os.path.join(aug_rc_lbl_dir, lbl_name))
        val_rc_nrcbkg_img_txt.write("%s\n" % f)
        val_rc_nrcbkg_lbl_txt.write("%s\n" % os.path.join(aug_rc_lbl_dir, lbl_name))
    val_rc_img_txt.close()
    val_rc_lbl_txt.close()

    non_rcids = rcids[rcids != rc_id]
    print('non_rcids', non_rcids)
    cnt = 0
    for nrc_id in non_rcids:
        aug_nrc_lbl_dir = os.path.join(args.annos_save_dir[:-1] + '_rc', 'rc{}_flip_aug'.format(nrc_id))
        aug_nrc_img_dir = os.path.join(args.images_save_dir[:-1] + '_rc', 'rc{}_flip_aug'.format(nrc_id))
        aug_nrc_img_files = np.sort(glob.glob(os.path.join(aug_nrc_img_dir, '*.jpg')))
        for nf in aug_nrc_img_files:
            val_rc_nrcbkg_img_txt.write("%s\n" % nf)
            img_name = os.path.basename(nf)
            lbl_name = img_name.replace('.jpg', '.txt')
            val_rc_nrcbkg_lbl_txt.write("%s\n" % os.path.join(aug_nrc_lbl_dir, lbl_name))
            cnt += 1
    print('non-rc cnt', cnt)

    val_cc1_img_dir = args.images_save_dir[:-1] + '_cc1_val'
    val_cc2_img_dir = args.images_save_dir[:-1] + '_cc2_val'
    val_cc1_lbl_dir =  args.annos_save_dir[:-1] + '_cc1_with_id_val'
    val_cc2_lbl_dir =  args.annos_save_dir[:-1] + '_cc2_with_id_val'
    val_cc1_lbl_files = np.sort(glob.glob(os.path.join(val_cc1_lbl_dir, '*.txt')))
    val_cc2_lbl_files = np.sort(glob.glob(os.path.join(val_cc2_lbl_dir, '*.txt')))
    print('val_cc1_lbl_files, val_cc2_lbl_files', len(val_cc1_lbl_files), len(val_cc2_lbl_files))
    cnt = 0
    for f in val_cc1_lbl_files:
        lbl_name = os.path.basename(f)
        if lbl_name in all_ori_lbl_names:
            continue
        val_rc_nrcbkg_lbl_txt.write("%s\n" % nf)
        img_name = lbl_name.replace('.txt', '.jpg')
        val_rc_nrcbkg_img_txt.write("%s\n" % os.path.join(val_cc1_img_dir, img_name))
        cnt += 1


    for f in val_cc2_lbl_files:
        lbl_name = os.path.basename(f)
        if lbl_name in all_ori_lbl_names:
            continue
        val_rc_nrcbkg_lbl_txt.write("%s\n" % nf)
        img_name = lbl_name.replace('.txt', '.jpg')
        val_rc_nrcbkg_img_txt.write("%s\n" % os.path.join(val_cc2_img_dir, img_name))
        cnt += 1
    print('val cc1 + cc2', cnt)

    val_bkg_lbl_dir = args.annos_save_dir[:-1] + '_val_bkg_lbl'
    val_bkg_img_dir = args.images_save_dir[:-1] + '_val_bkg_img'
    val_bkg_lbl_files = np.sort(glob.glob(os.path.join(val_bkg_lbl_dir, '*.txt')))
    print('val_bkg_lbl_files', len(val_bkg_lbl_files))

    for f in val_bkg_lbl_files:
        val_rc_nrcbkg_lbl_txt.write("%s\n" % f)
        img_name = os.path.basename(f).replace('.txt', '.jpg')
        val_rc_nrcbkg_img_txt.write("%s\n" % os.path.join(val_bkg_img_dir, img_name))

    val_rc_nrcbkg_img_txt.close()
    val_rc_nrcbkg_lbl_txt.close()


def create_val_xview_rc_nrcbkg_data(rc_id, data_name, seed=17):
    base_cmt = 'px{}whr{}_seed{}'.format(px_thres, whr_thres, seed)
    data_save_dir =  os.path.join(args.data_save_dir, base_cmt, 'RC')
    print('data_save_dir', data_save_dir)
    data_txt = open(os.path.join(data_save_dir, '{}_test_{}.data'.format(data_name, base_cmt)), 'w')
    data_txt.write(
        'valid={}\n'.format(os.path.join(data_save_dir, '{}_val_img_{}.txt'.format(data_name, base_cmt))))
    data_txt.write(
        'valid_label={}\n'.format(os.path.join(data_save_dir, '{}_val_lbl_{}.txt'.format(data_name, base_cmt))))

    data_txt.write('classes=%s\n' % str(args.class_num))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval=color\n')
    data_txt.close()


def get_args(px_thres=None, whr_thres=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--xview_yolo_dir", type=str, help="dir to xViewYOLO",
                        default='/media/lab/Yang/data/xView_YOLO/')

    parser.add_argument("--data_xview_dir", type=str, help="to save data files",
                        default='/media/lab/Yang/code/yolov3/data_xview/')

    parser.add_argument("--images_save_dir", type=str, help="to save chip trn val images files",
                        default='/media/lab/Yang/data/xView_YOLO/images/')

    # parser.add_argument("--images_save_dir", type=str, help="to save chip trn val images files",
    #                     default='/media/lab/Yang/data/xView_YOLO/images/{}_')

    parser.add_argument("--syn_data_dir", type=str, help="to syn data list files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_{}_cls/')

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
    remove other airplane labels ccid=None, rcid=None
    '''
    # label_cc_and_rc_remove_other_airp_labels()
    '''
    check bbox-with ccid or rcid
    '''
    # plot_bbox_on_images_of_rc_and_cc()

    '''
    split train:val = 80%:20%
    split CC(i) i={1, 2}
    RC(2) is a subset of CC(2), keep all patches of RC(2) in val of CC(2)
    combine all RC* together
    '''
    # ccids=[1, 2]
    # for ccid in ccids:
    #     split_cc_ncc_and_rc_by_cc_id(ccid, seed=17)

    '''
    add NCC, NRC to BG
    '''
    # add_ncc_nrc_to_bkg()

    '''
    split BKG images into train and val
    '''
    # comments = 'px{}whr{}_seed{}'
    # px_thres = 23
    # whr_thres = 3
    # seed = 17
    # split_bkg_into_train_val(comments, seed)


    '''
    augment RC
    left-right flip
    then rotate90, 180, 270
    '''
    # rcids= np.array([1,2,3,4,5])
    # for rc_id in rcids:
    #     augment_RC_by_id(rc_id)

    '''
    RC validation set
    val: aug RC(i) + Non-RC(i) + CC + BKG
    '''
    # seed = 17
    # base_pxwhrs = 'px{}whr{}_seed{}'.format(px_thres, whr_thres, seed)
    # rcids = np.array([1,2,3,4,5])
    # for rc_id in rcids:
    #     combine_val_for_each_aug_RC_step_by_step(rc_id, base_pxwhrs, rcids)
    #     data_name = 'xview_oa_bkg_aug_rc{}'.format(rc_id)
    #     create_val_xview_rc_nrcbkg_data(rc_id, data_name, seed)

    '''
    split CC1, CC2 separately
    train: CC(i) + Non-CC(i) + BKG 
    val: CC(i) + Non-CC(i) + BKG + RC
    '''
    # seed = 17
    # base_pxwhrs = 'px{}whr{}_seed{}'.format(px_thres, whr_thres, seed)
    # ccids= np.array([1, 2])
    # for cc_id in ccids:
    #     combine_trn_val_for_each_CC_step_by_step(cc_id, base_pxwhrs, ccids)
    #     data_name = 'xview_oa_bkg_cc{}'.format(cc_id)
    #     create_xview_cc_nccbkg_data(cc_id, data_name, seed)

    '''
    combine syn and xview
    '''
    seed=17
    base_cmt='px{}whr{}_seed{}'.format(px_thres, whr_thres, seed)
    comments = []

    for cmt in comments:
        combine_syn_xview_cc(cmt, seed, base_cmt, bkg_cc_sep=False)


