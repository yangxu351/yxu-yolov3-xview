import os
import argparse
import glob
import shutil
import numpy as np
import pandas as pd



def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def split_trn_val_with_rc_step_by_step(data_name='xview', comments='', seed=17, px_thres=None, whr_thres=None):
    '''
    first step: split data contains aircrafts but no rc images
    second step: split data contains no aircrafts (bkg images)
    third step: split RC images train:val targets ~ 1:1 !!!! manully split
    ###################### important!!!!! the difference between '*.txt' and '_*.txt'
    ######################  set(l) will change the order of list
                           list(dict.fromkeys(l)) doesn't change the order of list
    '''
    args = get_args(px_thres, whr_thres)

    # import random
    # random.seed(seed)

    data_save_dir = args.data_save_dir
    if comments:
        txt_save_dir = args.data_list_save_dir + comments[1:] # + '_bh'+ '/'
        if not os.path.exists(txt_save_dir):
        #     shutil.rmtree(txt_save_dir)
        #     os.makedirs(txt_save_dir)
        # else:
            os.makedirs(txt_save_dir)
        data_save_dir = os.path.join(data_save_dir, comments[1:])
        if not os.path.exists(data_save_dir):
        #     shutil.rmtree(data_save_dir)
        #     os.makedirs(data_save_dir)
        # else:
            os.makedirs(data_save_dir)
    else:
        txt_save_dir = args.data_list_save_dir

    lbl_path = args.annos_save_dir
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
    # print('lbl_path', lbl_path)
    print('all_rc_img_names', len(all_rc_img_names))

    # print('trn_rc_lbl_files', trn_rc_lbl_files)
    trn_rc_img_files = [f for f in glob.glob(os.path.join(trn_rc_img_dir, '*.jpg'))]
    val_rc_img_files = [f for f in glob.glob(os.path.join(val_rc_img_dir, '*.jpg'))]
    print('trn_rc_img_files', len(trn_rc_img_files))
    print('val_rc_img_files', len(val_rc_img_files))
    trn_rc_lbl_files = [os.path.join(lbl_path, os.path.basename(f).replace('.jpg', '.txt')) for f in trn_rc_img_files]
    val_rc_lbl_files = [os.path.join(lbl_path, os.path.basename(f).replace('.jpg', '.txt')) for f in val_rc_img_files]


    # print('trn_rc_lbl_files', trn_rc_lbl_files)

    ##### images that contain aircrafts
    airplane_lbl_files = [f for f in glob.glob(os.path.join(lbl_path, '*.txt')) if is_non_zero_file(f)]
    airplane_lbl_files.sort()
    num_air_files = len(airplane_lbl_files)
    print('num_air_files', num_air_files)

    ##### images that contain no aircrafts (drop out by rules)
    airplane_ept_lbl_files = [os.path.join(lbl_path, os.path.basename(f)) for f in glob.glob(os.path.join(lbl_path, '*.txt')) if not is_non_zero_file(f)]
    airplane_ept_img_files = [os.path.join(images_save_dir, os.path.basename(f).replace('.txt', '.jpg')) for f in airplane_ept_lbl_files]
    print('airplane_ept_img_files', len(airplane_ept_img_files))

    ##### images that contain no aircrafts-- BKG
    bkg_lbl_files = glob.glob(os.path.join(bkg_lbl_dir, '*.txt'))
    bkg_lbl_files.sort()
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

    bkg_ixes = np.random.permutation(len(bkg_lbl_files))
    trn_non_bkg_num = len(trn_nrc_lbl_files) + len(trn_rc_lbl_files)
    val_non_bkg_num = len(val_rc_lbl_files) + len(val_rc_lbl_files)
    trn_bkg_lbl_files =[bkg_lbl_files[i] for i in bkg_ixes[:trn_non_bkg_num ]]
    val_bkg_lbl_files = [bkg_lbl_files[i] for i in bkg_ixes[ trn_non_bkg_num: val_non_bkg_num + trn_non_bkg_num]]
    trn_bkg_img_files = [bkg_img_files[i] for i in bkg_ixes[: trn_non_bkg_num]]
    val_bkg_img_files = [bkg_img_files[i] for i in bkg_ixes[trn_non_bkg_num: val_non_bkg_num + trn_non_bkg_num]]
    print('trn_bkg_lbl_files', len(trn_bkg_lbl_files), len(trn_bkg_img_files))

    # exit(0)

    trn_lbl_files = trn_bkg_lbl_files + trn_nrc_lbl_files + trn_rc_lbl_files
    val_lbl_files = val_bkg_lbl_files + val_nrc_lbl_files + val_rc_lbl_files
    trn_img_files = trn_bkg_img_files + trn_nrc_img_files + trn_rc_img_files
    val_img_files = val_bkg_img_files + val_nrc_img_files + val_rc_img_files

    print('trn_num ', len(trn_lbl_files), len(trn_img_files))
    print('val_num ', len(val_lbl_files), len(val_img_files))

    trn_img_txt = open(os.path.join(txt_save_dir, '{}train_img{}.txt'.format(data_name, comments)), 'w')
    trn_lbl_txt = open(os.path.join(txt_save_dir, '{}train_lbl{}.txt'.format(data_name, comments)), 'w')
    val_img_txt = open(os.path.join(txt_save_dir, '{}val_img{}.txt'.format(data_name, comments)), 'w')
    val_lbl_txt = open(os.path.join(txt_save_dir, '{}val_lbl{}.txt'.format(data_name, comments)), 'w')

    trn_lbl_dir = args.data_list_save_dir + comments[1:] + '_trn_lbl'
    val_lbl_dir = args.data_list_save_dir + comments[1:] + '_val_lbl'
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
        lbl_name = os.path.basename(trn_lbl_files[i])
        trn_img_txt.write("%s\n" % trn_img_files[i])
        shutil.copy(trn_lbl_files[i], os.path.join(trn_lbl_dir, lbl_name))
    trn_img_txt.close()
    trn_lbl_txt.close()

    for j in range(len(val_lbl_files)):
        # print('val_lbl_files ', j, val_lbl_files[j])
        val_lbl_txt.write("%s\n" % val_lbl_files[j])
        lbl_name = os.path.basename(val_lbl_files[j])
        val_img_txt.write("%s\n" % val_img_files[j])
        shutil.copy(val_lbl_files[j], os.path.join(val_lbl_dir, lbl_name))
        # exit(0)
    val_img_txt.close()
    val_lbl_txt.close()

    shutil.copyfile(os.path.join(txt_save_dir, '{}train_img{}.txt'.format(data_name, comments)),
                    os.path.join(data_save_dir, '{}train_img{}.txt'.format(data_name, comments)))
    shutil.copyfile(os.path.join(txt_save_dir, '{}train_lbl{}.txt'.format(data_name, comments)),
                    os.path.join(data_save_dir, '{}train_lbl{}.txt'.format(data_name, comments)))
    shutil.copyfile(os.path.join(txt_save_dir, '{}val_img{}.txt'.format(data_name, comments)),
                    os.path.join(data_save_dir, '{}val_img{}.txt'.format(data_name, comments)))
    shutil.copyfile(os.path.join(txt_save_dir, '{}val_lbl{}.txt'.format(data_name, comments)),
                    os.path.join(data_save_dir, '{}val_lbl{}.txt'.format(data_name, comments)))


def split_trn_val_nrc_bkg_with_rc_sep_step_by_step(data_name='xview_nrcbkg', comments='', seed=17, px_thres=None, whr_thres=None):
    '''
    first step: split data contains aircrafts but no rc images
    second step: split data contains no aircrafts (bkg images)
    third step: split RC images train:val targets ~ 1:1 !!!! manully split save rc training files separately
    ###################### important!!!!! the difference between '*.txt' and '_*.txt'
    ######################  set(l) will change the order of list
                           list(dict.fromkeys(l)) doesn't change the order of list
    '''
    args = get_args(px_thres, whr_thres)

    data_save_dir = args.data_save_dir
    if comments:
        txt_save_dir = args.data_list_save_dir + comments[1:] # + '_bh'+ '/'
        if not os.path.exists(txt_save_dir):
        #     shutil.rmtree(txt_save_dir)
        #     os.makedirs(txt_save_dir)
        # else:
            os.makedirs(txt_save_dir)
        data_save_dir = os.path.join(data_save_dir, comments[1:])
        if not os.path.exists(data_save_dir):
        #     shutil.rmtree(data_save_dir)
        #     os.makedirs(data_save_dir)
        # else:
            os.makedirs(data_save_dir)
    else:
        txt_save_dir = args.data_list_save_dir

    lbl_path = args.annos_save_dir
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
    # print('lbl_path', lbl_path)
    print('all_rc_img_names', len(all_rc_img_names))

    # print('trn_rc_lbl_files', trn_rc_lbl_files)
    trn_rc_img_files = [f for f in glob.glob(os.path.join(trn_rc_img_dir, '*.jpg'))]
    val_rc_img_files = [f for f in glob.glob(os.path.join(val_rc_img_dir, '*.jpg'))]
    print('trn_rc_img_files', len(trn_rc_img_files))
    print('val_rc_img_files', len(val_rc_img_files))
    trn_rc_lbl_files = [os.path.join(lbl_path, os.path.basename(f).replace('.jpg', '.txt')) for f in trn_rc_img_files]
    val_rc_lbl_files = [os.path.join(lbl_path, os.path.basename(f).replace('.jpg', '.txt')) for f in val_rc_img_files]

    trn_rc_img_txt = open(os.path.join(txt_save_dir, 'only_rc_train_img{}.txt'.format(comments)), 'w')
    trn_rc_lbl_txt = open(os.path.join(txt_save_dir, 'only_rc_train_lbl{}.txt'.format(comments)), 'w')
    for rf in trn_rc_img_files:
        trn_rc_img_txt.write('%s\n' % rf)
        trn_rc_lbl_txt.write('%s\n' % os.path.join(lbl_path, os.path.basename(rf).replace('.jpg', '.txt')))
    trn_rc_img_txt.close()
    trn_rc_lbl_txt.close()

    val_rc_img_txt = open(os.path.join(txt_save_dir, 'only_rc_val_img{}.txt'.format(comments)), 'w')
    val_rc_lbl_txt = open(os.path.join(txt_save_dir, 'only_rc_val_lbl{}.txt'.format(comments)), 'w')
    for rf in val_rc_img_files:
        val_rc_img_txt.write('%s\n' % rf)
        val_rc_lbl_txt.write('%s\n' % os.path.join(lbl_path, os.path.basename(rf).replace('.jpg', '.txt')))
    val_rc_img_txt.close()
    val_rc_lbl_txt.close()
    # print('trn_rc_lbl_files', trn_rc_lbl_files)
    shutil.copy(os.path.join(txt_save_dir, 'only_rc_train_img{}.txt'.format(comments)),
                    os.path.join(data_save_dir, 'only_rc_train_img{}.txt'.format(comments)))
    shutil.copy(os.path.join(txt_save_dir, 'only_rc_train_lbl{}.txt'.format(comments)),
                    os.path.join(data_save_dir, 'only_rc_train_lbl{}.txt'.format(comments)))
    shutil.copy(os.path.join(txt_save_dir, 'only_rc_val_img{}.txt'.format(comments)),
                    os.path.join(data_save_dir, 'only_rc_val_img{}.txt'.format(comments)))
    shutil.copy(os.path.join(txt_save_dir, 'only_rc_val_lbl{}.txt'.format(comments)),
                    os.path.join(data_save_dir, 'only_rc_val_lbl{}.txt'.format(comments)))

    ##### images that contain aircrafts (rc included)
    airplane_lbl_files = [f for f in glob.glob(os.path.join(lbl_path, '*.txt')) if is_non_zero_file(f)]
    airplane_lbl_files.sort()
    num_air_files = len(airplane_lbl_files)
    print('num_air_files', num_air_files)

    ##### images that contain no aircrafts (drop out by rules)
    airplane_ept_lbl_files = [os.path.join(lbl_path, os.path.basename(f)) for f in glob.glob(os.path.join(lbl_path, '*.txt')) if not is_non_zero_file(f)]
    airplane_ept_img_files = [os.path.join(images_save_dir, os.path.basename(f).replace('.txt', '.jpg')) for f in airplane_ept_lbl_files]
    print('airplane_ept_img_files', len(airplane_ept_img_files))

    ##### images that contain no aircrafts-- BKG
    bkg_lbl_files = glob.glob(os.path.join(bkg_lbl_dir, '*.txt'))
    bkg_lbl_files.sort()
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

    bkg_ixes = np.random.permutation(len(bkg_lbl_files))
    trn_non_bkg_num = len(trn_nrc_lbl_files)
    val_non_bkg_num = len(val_nrc_lbl_files)
    trn_bkg_lbl_files =[bkg_lbl_files[i] for i in bkg_ixes[:trn_non_bkg_num ]]
    val_bkg_lbl_files = [bkg_lbl_files[i] for i in bkg_ixes[ trn_non_bkg_num: val_non_bkg_num + trn_non_bkg_num]]
    trn_bkg_img_files = [bkg_img_files[i] for i in bkg_ixes[: trn_non_bkg_num]]
    val_bkg_img_files = [bkg_img_files[i] for i in bkg_ixes[trn_non_bkg_num: val_non_bkg_num + trn_non_bkg_num]]
    print('trn_bkg_lbl_files', len(trn_bkg_lbl_files), len(trn_bkg_img_files))

    # exit(0)

    trn_lbl_files = trn_bkg_lbl_files + trn_nrc_lbl_files
    val_lbl_files = val_bkg_lbl_files + val_nrc_lbl_files
    trn_img_files = trn_bkg_img_files + trn_nrc_img_files
    val_img_files = val_bkg_img_files + val_nrc_img_files

    print('trn_num ', len(trn_lbl_files), len(trn_img_files))
    print('val_num ', len(val_lbl_files), len(val_img_files))

    trn_img_txt = open(os.path.join(txt_save_dir, '{}_train_img{}.txt'.format(data_name, comments)), 'w')
    trn_lbl_txt = open(os.path.join(txt_save_dir, '{}_train_lbl{}.txt'.format(data_name, comments)), 'w')

    val_img_txt = open(os.path.join(txt_save_dir, 'xview_rc_nrcbkg_val_img{}.txt'.format(comments)), 'w')
    val_lbl_txt = open(os.path.join(txt_save_dir, 'xview_rc_nrcbkg_val_lbl{}.txt'.format(comments)), 'w')

    trn_lbl_dir = args.data_list_save_dir + comments[1:] + '_trn_lbl'
    val_lbl_dir = args.data_list_save_dir + comments[1:] + '_val_lbl'
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
        lbl_name = os.path.basename(trn_lbl_files[i])
        trn_img_txt.write("%s\n" % trn_img_files[i])
        shutil.copy(trn_lbl_files[i], os.path.join(trn_lbl_dir, lbl_name))
    trn_img_txt.close()
    trn_lbl_txt.close()

    for j in range(len(val_lbl_files)):
        # print('val_lbl_files ', j, val_lbl_files[j])
        val_lbl_txt.write("%s\n" % val_lbl_files[j])
        lbl_name = os.path.basename(val_lbl_files[j])
        val_img_txt.write("%s\n" % val_img_files[j])
        shutil.copy(val_lbl_files[j], os.path.join(val_lbl_dir, lbl_name))
        # exit(0)
    for j in range(len(val_rc_lbl_files)):
        val_lbl_txt.write("%s\n" % val_rc_lbl_files[j])
        lbl_name = os.path.basename(val_rc_lbl_files[j])
        val_img_txt.write("%s\n" % val_rc_img_files[j])
        shutil.copy(val_lbl_files[j], os.path.join(val_lbl_dir, lbl_name))
    val_img_txt.close()
    val_lbl_txt.close()

    shutil.copy(os.path.join(txt_save_dir, '{}_train_img{}.txt'.format(data_name, comments)),
                    os.path.join(data_save_dir, '{}_train_img{}.txt'.format(data_name, comments)))
    shutil.copy(os.path.join(txt_save_dir, '{}_train_lbl{}.txt'.format(data_name, comments)),
                    os.path.join(data_save_dir, '{}_train_lbl{}.txt'.format(data_name, comments)))


    shutil.copy(os.path.join(txt_save_dir, 'xview_rc_nrcbkg_val_img{}.txt'.format(comments)),
                    os.path.join(data_save_dir, 'xview_rc_nrcbkg_val_img{}.txt'.format(comments)))
    shutil.copy(os.path.join(txt_save_dir, 'xview_rc_nrcbkg_val_lbl{}.txt'.format(comments)),
                    os.path.join(data_save_dir, 'xview_rc_nrcbkg_val_lbl{}.txt'.format(comments)))


def create_xview_rc_nrcbkg_data(px_thres=23, whr_thres=3, seed=17, val_aug=True):
    args = get_args(px_thres, whr_thres)

    base_cmt = 'px{}whr{}_seed{}'.format(px_thres, whr_thres, seed)
    data_save_dir = args.data_save_dir
    print('data_save_dir', data_save_dir)
    if val_aug:
        data_txt = open(os.path.join(data_save_dir, base_cmt, 'xview_ori_nrcbkg_aug_rc_{}.data'.format(base_cmt)), 'w')
    else:
        data_txt = open(os.path.join(data_save_dir, base_cmt, 'xview_rc_nrcbkg_{}.data'.format(base_cmt)), 'w')
    data_txt.write(
        'xview_train={}\n'.format(os.path.join(data_save_dir, base_cmt, 'xview_nrcbkg_train_img_{}.txt'.format(base_cmt))))
    data_txt.write(
        'xview_train_label={}\n'.format(os.path.join(data_save_dir, base_cmt, 'xview_nrcbkg_train_lbl_{}.txt'.format(base_cmt))))

    data_txt.write(
        'rc_train={}\n'.format(os.path.join(data_save_dir, base_cmt,  'only_rc_train_img_{}.txt'.format(base_cmt))))
    data_txt.write(
        'rc_train_label={}\n'.format(os.path.join(data_save_dir,  base_cmt, 'only_rc_train_lbl_{}.txt'.format(base_cmt))))
    if val_aug:
        data_txt.write(
            'valid={}\n'.format(os.path.join(data_save_dir, base_cmt, 'xview_ori_nrcbkg_aug_rc_val_img_{}.txt'.format(base_cmt))))
        data_txt.write(
            'valid_label={}\n'.format(os.path.join(data_save_dir, base_cmt, 'xview_ori_nrcbkg_aug_rc_val_lbl_{}.txt'.format(base_cmt))))
    else:
        data_txt.write(
            'valid={}\n'.format(os.path.join(data_save_dir, base_cmt, 'xview_rc_nrcbkg_val_img_{}.txt'.format(base_cmt))))
        data_txt.write(
            'valid_label={}\n'.format(os.path.join(data_save_dir, base_cmt, 'xview_rc_nrcbkg_val_lbl_{}.txt'.format(base_cmt))))

    df_trn_nrcbkg = pd.read_csv(os.path.join(data_save_dir, base_cmt, 'xview_nrcbkg_train_img_{}.txt'.format(base_cmt)), header=None)
    df_trn_rc = pd.read_csv(os.path.join(data_save_dir, base_cmt,  'only_rc_train_img_{}.txt'.format(base_cmt)), header=None)
    xview_trn_num = df_trn_nrcbkg.shape[0] + df_trn_rc.shape[0]

    data_txt.write('syn_0_xview_number={}\n'.format(xview_trn_num))
    data_txt.write('classes=%s\n' % str(args.class_num))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval=color\n')
    data_txt.close()


def get_args(px_thres=None, whr_thres=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str,
                        help="Path to folder containing image chips (ie 'Image_Chips/') ",
                        default='/media/lab/Yang/data/xView/airplane_tifs/')

    parser.add_argument("--json_filepath", type=str, help="Filepath to GEOJSON coordinate file",
                        default='/media/lab/Yang/data/xView/xView_train.geojson')

    parser.add_argument("--xview_yolo_dir", type=str, help="dir to xViewYOLO",
                        default='/media/lab/Yang/data/xView_YOLO/')

    parser.add_argument("--images_save_dir", type=str, help="to save chip trn val images files",
                        default='/media/lab/Yang/data/xView_YOLO/images/')

    parser.add_argument("--txt_save_dir", type=str, help="to save  related label files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')

    parser.add_argument("--data_list_save_dir", type=str, help="to save selected trn val images and labels",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_cls/data_list/')

    parser.add_argument("--annos_save_dir", type=str, help="to save txt annotation files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')

    parser.add_argument("--data_save_dir", type=str, help="to save data files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_cls/')

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
    # args.images_save_dir = args.images_save_dir + '{}_{}cls_reduced/'.format(args.input_size, args.class_num)
    args.annos_new_dir = args.annos_save_dir + '{}/{}_cls_xcycwh_part_new/'.format(args.input_size, args.class_num)
    if px_thres:
        args.annos_save_dir = args.annos_save_dir + '{}/{}_cls_xcycwh_px{}whr{}/'.format(args.input_size, args.class_num, px_thres, whr_thres)
    else:
        args.annos_save_dir = args.annos_save_dir + '{}/{}_cls_xcycwh/'.format(args.input_size, args.class_num)
    args.txt_save_dir = args.txt_save_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)
    args.data_save_dir = args.data_save_dir.format(args.class_num)
    args.data_list_save_dir = args.data_list_save_dir.format(args.input_size, args.class_num)
    if not os.path.exists(args.txt_save_dir):
        os.makedirs(args.txt_save_dir)

    if not os.path.exists(args.annos_new_dir):
        os.makedirs(args.annos_new_dir)

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
    #fixme
    # split train val only nrc and bkg,
    # split rc separately
    px_thres = 23
    whr_thres = 3
    seed = 17
    comments = '_px{}whr{}_seed{}'.format(px_thres, whr_thres, seed)
    # data_name = 'xview_nrcbkg'
    # split_trn_val_nrc_bkg_with_rc_sep_step_by_step(data_name, comments, seed, px_thres, whr_thres)
    create_xview_rc_nrcbkg_data(px_thres, whr_thres, seed, val_aug=True)
