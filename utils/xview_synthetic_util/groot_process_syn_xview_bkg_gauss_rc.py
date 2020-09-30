'''
xview process
in order to generate xivew 2-d background with synthetic airplances
'''
import glob
import numpy as np
import argparse
import os
from PIL import Image
import pandas as pd
import shutil


IMG_FORMAT = '.png'
TXT_FORMAT = '.txt'



def split_syn_xview_background_trn_val(seed=17, comment='syn_xview_background_texture', pxwhr='', base_pxwhrs='px23whr3_seed17', upscale=False, display_type='color'):

    data_xview_dir = syn_args.data_xview_dir.format( syn_args.class_num)
    
    data_dir = syn_args.syn_data_list_dir.format(comment, syn_args.class_num, comment, seed)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    step = syn_args.tile_size * syn_args.resolution
    if upscale:
        all_files = np.sort(glob.glob(os.path.join(syn_args.syn_data_dir, 'upscale_{}_all_images_step{}'.format(display_type, step), '*' + IMG_FORMAT)))
        lbl_dir = os.path.join(syn_args.syn_annos_dir, 'minr{}_linkr{}_{}_upscale_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, pxwhr, display_type, step))

    else:
        print('img dir', os.path.join(syn_args.syn_data_dir, '{}_all_images_step{}'.format(display_type, step)))
        all_files = np.sort(glob.glob(os.path.join(syn_args.syn_data_dir, '{}_all_images_step{}'.format(display_type, step), '*' + IMG_FORMAT)))
        lbl_dir = os.path.join(syn_args.syn_annos_dir, 'minr{}_linkr{}_{}_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, pxwhr, display_type, step))
    print('img dir', os.path.join(syn_args.syn_data_dir, '{}_all_images_step{}'.format(display_type, step)))    
    trn_img_txt = open(os.path.join(data_dir, '{}_train_img_seed{}.txt'.format(comment, seed)), 'w')
    trn_lbl_txt = open(os.path.join(data_dir, '{}_train_lbl_seed{}.txt'.format(comment, seed)), 'w')
    val_img_txt = open(os.path.join(data_dir, '{}_val_img_seed{}.txt'.format(comment, seed)), 'w')
    val_lbl_txt = open(os.path.join(data_dir, '{}_val_lbl_seed{}.txt'.format(comment, seed)), 'w')
#    print(os.path.join(syn_args.syn_data_dir.format(display_type), '{}_all_images_step{}'.format(display_type, step), '*' + IMG_FORMAT))
    num_files = len(all_files)
    print('num_files', num_files)

    #fixme---yang.xu
    num_val = int(num_files*syn_args.val_percent)
    num_trn = num_files - num_val

    np.random.seed(seed)
    all_indices = np.random.permutation(num_files)
    print('num_trn', num_trn)
    for j in all_indices[: num_trn]:
#        print('all_files[i]', all_files[j])
        trn_img_txt.write('%s\n' % all_files[j])
        trn_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(all_files[j]).replace(IMG_FORMAT, TXT_FORMAT)))
    trn_img_txt.close()
    trn_lbl_txt.close()
    for i in all_indices[num_trn:num_trn+num_val ]:
        val_img_txt.write('%s\n' % all_files[i])
        val_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(all_files[i]).replace(IMG_FORMAT, TXT_FORMAT)))
    val_img_txt.close()
    val_lbl_txt.close()

def record_all_syn_xview_background(comment='syn_xview_background_texture', seed=1024, pxwhr='', upscale=False):
    '''
    no train val slit
    all syn images/labels
    :param comment:
    :param seed:
    :param pxwhr:
    :return:
    '''
    display_type = comment.split('_')[-1]
    data_dir = syn_args.syn_data_list_dir.format(comment, syn_args.class_num, comment, seed)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    step = syn_args.tile_size * syn_args.resolution
    if upscale:
        all_files = np.sort(glob.glob(os.path.join(syn_args.syn_data_dir, 'upscale_{}_all_images_step{}'.format(display_type, step), '*' + IMG_FORMAT)))
        lbl_dir = os.path.join(syn_args.syn_annos_dir, 'minr{}_linkr{}_{}_upscale_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, pxwhr, display_type, step))
        all_img_txt = open(os.path.join(data_dir, '{}_all_img_seed{}_upscale.txt'.format(comment, seed)), 'w')
        all_lbl_txt = open(os.path.join(data_dir, '{}_all_lbl_seed{}_upscale.txt'.format(comment, seed)), 'w')
    else:
        all_files = np.sort(glob.glob(os.path.join(syn_args.syn_data_dir, '{}_all_images_step{}'.format(display_type, step), '*' + IMG_FORMAT)))
        lbl_dir = os.path.join(syn_args.syn_annos_dir, 'minr{}_linkr{}_{}_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, pxwhr, display_type, step))
    all_img_txt = open(os.path.join(data_dir, '{}_all_img_seed{}.txt'.format(comment, seed)), 'w')
    all_lbl_txt = open(os.path.join(data_dir, '{}_all_lbl_seed{}.txt'.format(comment, seed)), 'w')

    for f in all_files:
        all_img_txt.write('%s\n' % f)
        all_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(f).replace(IMG_FORMAT, TXT_FORMAT)))
    all_img_txt.close()
    all_lbl_txt.close()

def record_target_neutral_of_model_id(pxwhr='px23whr3', base_pxwhrs='', model_id=None):
    annos_dir = syn_args.data_annos_dir[:-1] + '_{}_m{}_val_tgt_neu_with_modelid/'.format(pxwhr, model_id)
    all_files = glob.glob(os.path.join(annos_dir, '*.txt'))
    data_xview_dir = os.path.join(syn_args.data_xview_dir.format(syn_args.class_num), base_pxwhrs)
    all_img_txt = open(os.path.join(data_xview_dir, 'xviewtest_img_{}_m{}_tgt_neu.txt'.format(base_pxwhrs, model_id)), 'w')
    all_lbl_txt = open(os.path.join(data_xview_dir, 'xviewtest_lbl_{}_with_model_m{}_tgt_neu.txt'.format(base_pxwhrs, model_id)), 'w')
    img_dir = syn_args.data_img_dir
    for f in all_files:
        all_lbl_txt.write('%s\n' % f)
        all_img_txt.write('%s\n' % os.path.join(img_dir, os.path.basename(f).replace(TXT_FORMAT, '.jpg')))
    all_img_txt.close()
    all_lbl_txt.close()

    data_txt = open(os.path.join(data_xview_dir, 'xviewtest_{}_with_model_m{}_tgt_neu.data'.format(base_pxwhrs, model_id)), 'w')
    data_txt.write('test=./data_xview/{}_cls/{}/xviewtest_img_{}_m{}_tgt_neu.txt\n'.format(syn_args.class_num, base_pxwhrs, base_pxwhrs, label_id))
    data_txt.write('test_label=./data_xview/{}_cls/{}/xviewtest_lbl_{}_with_model_m{}_tgt_neu.txt\n'.format(syn_args.class_num, base_pxwhrs, base_pxwhrs, label_id))
    data_txt.write('classes=%s\n' % str(syn_args.class_num))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(syn_args.class_num))
    data_txt.close()
    
def create_syn_data(comment='syn_xview_background_texture', seed=1024, base_pxwhrs='px23whr3_seed17', val_xview=False, miss_id=None, label_id=None, upscale=False):
    data_dir = syn_args.syn_data_list_dir.format(comment, syn_args.class_num, comment, seed)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    dt = comment.split('_')[-1]
    if val_xview:
        if miss_id:
            data_txt = open(os.path.join(data_dir, '{}_seed{}_xview_val_labeled_miss.data'.format(comment, seed)), 'w')
        elif label_id:
            data_txt = open(os.path.join(data_dir, '{}_seed{}_xview_val_labeled.data'.format(comment, seed)), 'w')
        else:
            data_txt = open(os.path.join(data_dir, '{}_seed{}_xview_val.data'.format(comment, seed)), 'w')
    else:
        print('create *.data')
        data_txt = open(os.path.join(data_dir, '{}_seed{}.data'.format(comment, seed)), 'w')
        data_txt.write('train=./data_xview/{}_{}_cls/{}_seed{}/{}_train_img_seed{}.txt\n'.format( comment, syn_args.class_num, comment, seed, comment, seed))
        data_txt.write('train_label=./data_xview/{}_{}_cls/{}_seed{}/{}_train_lbl_seed{}.txt\n'.format(comment, syn_args.class_num, comment, seed, comment, seed))

     #fixme **********
    data_xview_dir = syn_args.data_xview_dir.format(syn_args.class_num)
    df = pd.read_csv(os.path.join(data_xview_dir, base_pxwhrs, 'xviewtrain_img_{}.txt'.format(base_pxwhrs)), header=None)
    data_txt.write('syn_0_xview_number={}\n'.format(df.shape[0]*4 if upscale else df.shape[0]))
    data_txt.write('classes=%s\n' % str(syn_args.class_num))

    if val_xview:
        if miss_id:
            data_txt.write('valid=./data_xview/{}_cls/{}/xviewtest_img_{}_m{}_labeled_miss.txt\n'.format(syn_args.class_num, base_pxwhrs, base_pxwhrs, miss_id))
            data_txt.write('valid_label=./data_xview/{}_cls/{}/xviewtest_lbl_{}_with_model_m{}_labeled_miss.txt\n'.format(syn_args.class_num, base_pxwhrs, base_pxwhrs, miss_id))
        elif label_id:
            data_txt.write('valid=./data_xview/{}_cls/{}/xviewtest_img_{}_m{}_labeled.txt\n'.format(syn_args.class_num, base_pxwhrs, base_pxwhrs, label_id))
            data_txt.write('valid_label=./data_xview/{}_cls/{}/xviewtest_lbl_{}_with_model_m{}_labeled.txt\n'.format(syn_args.class_num, base_pxwhrs, base_pxwhrs, label_id))
        else:
            data_txt.write('valid=./data_xview/{}_cls/{}/xviewval_img_{}.txt\n'.format(syn_args.class_num, base_pxwhrs, base_pxwhrs))
            data_txt.write('valid_label=./data_xview/{}_cls/{}/xviewval_lbl_{}_with_model.txt\n'.format(syn_args.class_num, base_pxwhrs, base_pxwhrs))
    else:
#        if upscale:
#            data_txt.write('valid=./data_xview/{}_{}_cls/{}_seed{}/{}_val_img_seed{}_upscale.txt\n'.format(comment, syn_args.class_num, comment, seed, comment, seed))
#            data_txt.write('valid_label=./data_xview/{}_{}_cls/{}_seed{}/{}_val_lbl_seed{}_upscale.txt\n'.format(comment, syn_args.class_num, comment, seed, comment, seed))
#        else:
        data_txt.write('valid=./data_xview/{}_{}_cls/{}_seed{}/{}_val_img_seed{}.txt\n'.format(comment, syn_args.class_num, comment, seed, comment, seed))
        data_txt.write('valid_label=./data_xview/{}_{}_cls/{}_seed{}/{}_val_lbl_seed{}.txt\n'.format(comment, syn_args.class_num, comment, seed, comment, seed))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(syn_args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval={}'.format(dt))
    data_txt.close()


def create_syn_data_with_model(comment='syn_xview_background_texture', seed=1024, base_cmt='', miss_id=None, label_id=None):
    data_dir = syn_args.syn_data_list_dir.format( comment, syn_args.class_num, comment, seed)
    dt = comment.split('_')[-1]

    if miss_id:
        data_txt = open(os.path.join(data_dir, '{}_seed{}_with_model_miss.data'.format(comment, seed)), 'w')
        data_txt.write('valid=./data_xview/{}_cls/{}_seed{}/xviewtest_img_{}_seed{}_m{}_miss.txt\n'.format(syn_args.class_num, base_cmt, seed, base_cmt, seed, miss_id))
        data_txt.write('valid_label=./data_xview/{}_cls/{}_seed{}/xviewtest_lbl_{}_seed{}_with_model_m{}_miss.txt\n'.format(syn_args.class_num, base_cmt, seed, base_cmt, seed, miss_id))
    if label_id:
        data_txt = open(os.path.join(data_dir, '{}_seed{}_with_model_labeled.data'.format(comment, seed)), 'w')
        data_txt.write('valid=./data_xview/{}_cls/{}_seed{}/xviewtest_img_{}_seed{}_m{}_labeled.txt\n'.format(syn_args.class_num, base_cmt, seed, base_cmt, seed, label_id))
        data_txt.write('valid_label=./data_xview/{}_cls/{}_seed{}/xviewtest_lbl_{}_seed{}_with_model_m{}_labeled.txt\n'.format(syn_args.class_num, base_cmt, seed, base_cmt, seed, label_id))
    else:
        data_txt = open(os.path.join(data_dir, '{}_seed{}_with_model.data'.format(comment, seed)), 'w')
        data_txt.write('valid=./data_xview/{}_cls/{}_seed{}/xviewval_img_{}_seed{}.txt\n'.format(syn_args.class_num, base_cmt, seed, base_cmt, seed))
        data_txt.write('valid_label=./data_xview/{}_cls/{}_seed{}/xviewval_lbl_{}_seed{}_with_model.txt\n'.format(syn_args.class_num, base_cmt, seed, base_cmt, seed))

    data_txt.write('classes=%s\n' % str(syn_args.class_num))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(syn_args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval={}'.format(dt))
    data_txt.close()


def split_syn_xview_background_trn_val_of_ratios(ratios, seed=1024, comment='xview_syn_xview_bkg_texture', base_cmt='px6whr4_ng0'):
    display_type = comment.split('_')[-1]
    data_dir = os.path.join(syn_args.data_xview_dir.format(syn_args.class_num), '{}_seed{}'.format(comment, seed))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    step = syn_args.tile_size * syn_args.resolution

    all_files = np.sort(glob.glob(os.path.join(syn_args.syn_data_dir, '{}_all_images_step{}'.format(display_type, step), '*' + IMG_FORMAT)))
    num_files = len(all_files)

    base_trn_img = np.loadtxt(os.path.join(syn_args.data_xview_dir, '{}_seed{}/xviewtrain_img_{}_seed{}.txt'.format(base_cmt, seed, base_cmt, seed)), dtype='str')
    base_trn_lbl = np.loadtxt(os.path.join(syn_args.data_xview_dir, '{}_seed{}/xviewtrain_lbl_{}_seed{}.txt'.format(base_cmt, seed, base_cmt, seed)), dtype='str')
    num_trn_base = base_trn_img.shape[0]

    # base_val_img = np.loadtxt(os.path.join(syn_args.data_xview_dir, 'px6whr4_ng0_seed{}/xviewval_img_px6whr4_ng0_seed{}.txt'.format(seed, seed)), dtype='str')
    # base_val_lbl = np.loadtxt(os.path.join(syn_args.data_xview_dir, 'px6whr4_ng0_seed{}/xviewval_lbl_px6whr4_ng0_seed{}.txt'.format(seed, seed)), dtype='str')
    # num_val_base = base_val_img.shape[0]

    np.random.seed(seed)
    all_indices = np.random.permutation(num_files)

    for r in ratios:
        num_trn_syn = num_trn_base * r
        trn_img_txt = open(os.path.join(data_dir, '{}_train_img_seed{}_{}xSyn.txt'.format(comment, seed, r)), 'w')
        trn_lbl_txt = open(os.path.join(data_dir, '{}_train_lbl_seed{}_{}xSyn.txt'.format(comment, seed, r)), 'w')
        for i in range(base_trn_img.size):
            trn_img_txt.write('%s\n' % base_trn_img[i])
            trn_lbl_txt.write('%s\n' % base_trn_lbl[i])

        lbl_dir = os.path.join(syn_args.syn_annos_dir, 'minr{}_linkr{}_{}_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, base_cmt, display_type, step))
        for j in all_indices[:num_trn_syn]:
            trn_img_txt.write('%s\n' % all_files[j])
            trn_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(all_files[j]).replace(IMG_FORMAT, TXT_FORMAT)))
        trn_img_txt.close()
        trn_lbl_txt.close()


def create_xview_syn_data(comment='xview_syn_xview_bkg_texture', seed=17, base_cmt='px6whr4_ng0_seed17', ratio=0, model_miss=None):
    dt = comment.split('_')[-1]
    data_dir = os.path.join(syn_args.data_xview_dir.format(syn_args.class_num), '{}_seed{}'.format(comment, seed))
    if ratio:
        if model_miss:
            data_txt = open(os.path.join(data_dir, '{}_seed{}_{}xSyn_miss.data'.format(comment, seed, ratio)), 'w')
        else:
            data_txt = open(os.path.join(data_dir, '{}_seed{}_{}xSyn.data'.format(comment, seed, ratio)), 'w')
        data_txt.write('train=./data_xview/{}_cls/{}_seed{}/{}_train_img_seed{}_{}xSyn.txt\n'.format(syn_args.class_num, comment, seed, comment, seed, ratio))
        data_txt.write('train_label=./data_xview/{}_cls/{}_seed{}/{}_train_lbl_seed{}_{}xSyn.txt\n'.format(syn_args.class_num, comment, seed, comment, seed, ratio))
    else:
        data_txt = open(os.path.join(data_dir, '{}_seed{}.data'.format(comment, seed)), 'w')
        data_txt.write('xview_train=./data_xview/{}_cls/{}/xviewtrain_img_{}.txt\n'.format(syn_args.class_num, base_cmt, base_cmt))
        data_txt.write('xview_train_label=./data_xview/{}_cls/{}/xviewtrain_lbl_{}.txt\n'.format(syn_args.class_num, base_cmt, base_cmt))
        if comment.startswith('xview_'):
            syn_cmt = comment[6:]
        data_txt.write('syn_train=./data_xview/{}_{}_cls/{}_seed{}/{}_all_img_seed{}.txt\n'.format(syn_cmt, syn_args.class_num, syn_cmt, seed, syn_cmt, seed))
        data_txt.write('syn_train_label=./data_xview/{}_{}_cls/{}_seed{}/{}_all_lbl_seed{}.txt\n'.format(syn_cmt, syn_args.class_num, syn_cmt, seed, syn_cmt, seed))

     #fixme **********
    xview_dir = os.path.join(syn_args.data_xview_dir.format(syn_args.class_num), '{}'.format(base_cmt))
    df = pd.read_csv(os.path.join(xview_dir, 'xviewtrain_img_{}.txt'.format(base_cmt)), header=None) # **********
    data_txt.write('syn_0_xview_number={}\n'.format(df.shape[0]))
    data_txt.write('classes=%s\n' % str(syn_args.class_num))
    # os.path.join(syn_args.data_xview_dir, '{}/xviewval_img_{}.txt'.format(base_cmt,base_cmt))
    if model_miss:
        data_txt.write('valid=./data_xview/{}_cls/{}/xviewval_img_{}_{}_miss.txt\n'.format(syn_args.class_num, base_cmt,  base_cmt, model_miss))
        data_txt.write('valid_label=./data_xview/{}_cls/{}/xviewval_lbl_{}_{}_miss.txt\n'.format(syn_args.class_num, base_cmt, base_cmt, model_miss))
    else:
        data_txt.write('valid=./data_xview/{}_cls/{}/xviewval_img_{}.txt\n'.format(syn_args.class_num, base_cmt,  base_cmt))
        data_txt.write('valid_label=./data_xview/{}_cls/{}/xviewval_lbl_{}.txt\n'.format(syn_args.class_num, base_cmt, base_cmt))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(syn_args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval={}'.format(dt))
    data_txt.close()


def create_xview_syn_data_with_model(comment='xview_syn_xview_bkg_texture', seed=17, base_cmt='px23whr3_seed17', ratio=0):
    dt = comment.split('_')[-1]
    data_dir = os.path.join(syn_args.data_xview_dir.format(syn_args.class_num), '{}_seed{}'.format(comment, seed))
    if ratio:
        data_txt = open(os.path.join(data_dir, '{}_seed{}_{}xSyn_with_model.data'.format(comment, seed, ratio)), 'w')
    else:
        data_txt = open(os.path.join(data_dir, '{}_seed{}_with_model.data'.format(comment, seed)), 'w')

    data_txt.write('classes=%s\n' % str(syn_args.class_num))
    os.path.join(syn_args.data_xview_dir, '{}/xviewval_img_{}.txt'.format(base_cmt,base_cmt))
    data_txt.write('valid=./data_xview/{}_cls/{}/xviewval_img_{}.txt\n'.format(syn_args.class_num, base_cmt, base_cmt))
    data_txt.write('valid_label=./data_xview/{}_cls/{}/xviewval_lbl_{}_with_model.txt\n'.format(syn_args.class_num, base_cmt, base_cmt))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(syn_args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval={}'.format(dt))
    data_txt.close()


def create_xview_base_data_for_onemodel_only(model_id, base_cmt='px23whr3_seed17'):
    syn_args = get_syn_args()
    data_dir = os.path.join(syn_args.data_xview_dir.format(syn_args.class_num), base_cmt)

    data_withmodel_txt = open(os.path.join(data_dir, 'xviewtest_{}_with_model_m{}_only.data'.format(base_cmt, model_id)), 'w')
    # data_withmodel_txt.write('xview_train=./data_xview/{}_cls/{}/xviewtrain_img_{}.txt\n'.format(syn_args.class_num, base_cmt, base_cmt))
    # data_withmodel_txt.write('xview_train_label=./data_xview/{}_cls/{}/xviewtrain_lbl_{}.txt\n'.format(syn_args.class_num, base_cmt, base_cmt))
    data_withmodel_txt.write('classes=%s\n' % str(syn_args.class_num))
    data_withmodel_txt.write('test=./data_xview/{}_cls/{}/xviewtest_img_{}_m{}_only.txt\n'.format(syn_args.class_num, base_cmt, base_cmt, model_id))
    data_withmodel_txt.write('test_label=./data_xview/{}_cls/{}/xviewtest_lbl_{}_with_model_m{}_only.txt\n'.format(syn_args.class_num, base_cmt, base_cmt, model_id))
    data_withmodel_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(syn_args.class_num))
    data_withmodel_txt.write('backup=backup/\n')
    data_withmodel_txt.close()

    data_txt = open(os.path.join(data_dir, 'xview_{}_m{}_only.data'.format(base_cmt, model_id)), 'w')
    data_txt.write('xview_train=./data_xview/{}_cls/{}/xviewtrain_img_{}.txt\n'.format(syn_args.class_num, base_cmt, base_cmt))
    data_txt.write('xview_train_label=./data_xview/{}_cls/{}/xviewtrain_lbl_{}.txt\n'.format(syn_args.class_num, base_cmt, base_cmt))
    data_txt.write('classes=%s\n' % str(syn_args.class_num))
    data_txt.write('valid=./data_xview/{}_cls/{}/xviewtest_img_{}_m{}_only.txt\n'.format(syn_args.class_num, base_cmt, base_cmt, model_id))
    data_txt.write('valid_label=./data_xview/{}_cls/{}/xviewtest_lbl_{}_with_model_m{}_only.txt\n'.format(syn_args.class_num, base_cmt, base_cmt, model_id))
    data_txt.write('test=./data_xview/{}_cls/{}/xviewtest_img_{}_m{}_only.txt\n'.format(syn_args.class_num, base_cmt, base_cmt, model_id))
    data_txt.write('test_label=./data_xview/{}_cls/{}/xviewtest_lbl_{}_with_model_m{}_only.txt\n'.format(syn_args.class_num, base_cmt, base_cmt, model_id))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(syn_args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.close()
    
def create_xview_base_data_for_onemodel_easy_hard(model_id, rc_id, eh_type='hard', base_cmt='px23whr3_seed17'):
    syn_args = get_syn_args()
    data_dir = os.path.join(syn_args.data_xview_dir.format(syn_args.class_num), base_cmt)
    data_withmodel_txt = open(os.path.join(data_dir, 'xviewtest_{}_m{}_rc{}_{}_aug.data'.format(base_cmt, model_id, rc_id, eh_type)), 'w')
    data_withmodel_txt.write('classes=%s\n' % str(syn_args.class_num))
    data_withmodel_txt.write('test=./data_xview/{}_cls/{}/xviewtest_img_{}_m{}_rc{}_{}_aug.txt\n'.format(syn_args.class_num, base_cmt, base_cmt, model_id, rc_id, eh_type))
    data_withmodel_txt.write('test_label=./data_xview/{}_cls/{}/xviewtest_lbl_{}_m{}_rc{}_{}_aug.txt\n'.format(syn_args.class_num, base_cmt, base_cmt, model_id, rc_id, eh_type))
    data_withmodel_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(syn_args.class_num))
    data_withmodel_txt.write('backup=backup/\n')
    data_withmodel_txt.close()


def create_xview_bkg_and_syn_data(comment='xview_syn_xview_bkg_texture', seed=17, base_cmt='px23whr3_seed17', model_miss=None):
    dt = comment.split('_')[-1]
    data_dir = os.path.join(syn_args.data_xview_dir.format(syn_args.class_num), '{}_seed{}'.format(comment, seed))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    data_txt = open(os.path.join(data_dir, '{}_seed{}_xbkgonly.data'.format(comment, seed)), 'w')
    bkg_cmt='xview_bkg_only_seed17'
    data_txt.write('xview_train=./data_xview/{}_cls/{}/xview_bkg_img_seed{}.txt\n'.format(syn_args.class_num, bkg_cmt, sd))
    data_txt.write('xview_train_label=./data_xview/{}_cls/{}/xview_bkg_lbl_seed{}.txt\n'.format(syn_args.class_num, bkg_cmt, sd))
    if comment.startswith('xview_'):
        syn_cmt = comment[6:]
    data_txt.write('syn_train=./data_xview/{}_{}_cls/{}_seed{}/{}_all_img_seed{}.txt\n'.format(syn_cmt, syn_args.class_num, syn_cmt, seed, syn_cmt, seed))
    data_txt.write('syn_train_label=./data_xview/{}_{}_cls/{}_seed{}/{}_all_lbl_seed{}.txt\n'.format(syn_cmt, syn_args.class_num, syn_cmt, seed, syn_cmt, seed))

     #fixme **********
    xview_dir = os.path.join(syn_args.data_xview_dir.format(syn_args.class_num), '{}'.format(base_cmt))
    df = pd.read_csv(os.path.join(xview_dir, 'xviewtrain_img_{}.txt'.format(base_cmt)), header=None) # **********
    data_txt.write('syn_0_xview_number={}\n'.format(df.shape[0]))
    data_txt.write('classes=%s\n' % str(syn_args.class_num))
    # os.path.join(syn_args.data_xview_dir, '{}/xviewval_img_{}.txt'.format(base_cmt,base_cmt))
    if model_miss:
        data_txt.write('valid=./data_xview/{}_cls/{}/xviewval_img_{}_{}_miss.txt\n'.format(syn_args.class_num, base_cmt,  base_cmt, model_miss))
        data_txt.write('valid_label=./data_xview/{}_cls/{}/xviewval_lbl_{}_{}_miss.txt\n'.format(syn_args.class_num, base_cmt, base_cmt, model_miss))
    else:
        data_txt.write('valid=./data_xview/{}_cls/{}/xviewval_img_{}.txt\n'.format(syn_args.class_num, base_cmt,  base_cmt))
        data_txt.write('valid_label=./data_xview/{}_cls/{}/xviewval_lbl_{}.txt\n'.format(syn_args.class_num, base_cmt, base_cmt))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(syn_args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval={}'.format(dt))
    data_txt.close()


def create_xview_base_data_for_onemodel_aug_easy_hard(model_id, rc_id, eh_type='hard', base_cmt='px23whr3_seed17'):
    syn_args = get_syn_args()
    data_dir = os.path.join(syn_args.data_xview_dir.format(syn_args.class_num), base_cmt)
    data_withmodel_txt = open(os.path.join(data_dir,
                                           'xviewtest_{}_m{}_rc{}_{}_aug.data'.format(base_cmt, model_id,
                                                                                                 rc_id, eh_type)), 'w')
    data_withmodel_txt.write('classes=%s\n' % str(syn_args.class_num))
    data_withmodel_txt.write(
        'test=./data_xview/{}_cls/{}/xviewtest_img_{}_m{}_rc{}_{}_aug.txt\n'.format(syn_args.class_num, base_cmt,
                                                                                    base_cmt, model_id, rc_id, eh_type))
    data_withmodel_txt.write(
        'test_label=./data_xview/{}_cls/{}/xviewtest_lbl_{}_m{}_rc{}_{}_aug.txt\n'.format(syn_args.class_num, base_cmt,
                                                                                          base_cmt, model_id, rc_id,
                                                                                          eh_type))
    data_withmodel_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(syn_args.class_num))
    data_withmodel_txt.write('backup=backup/\n')
    data_withmodel_txt.close()
    

def get_syn_args(cmt='certain_models'):
    parser = argparse.ArgumentParser()

    if cmt:
        parser.add_argument("--syn_data_dir", type=str,
                        default='/data/users/yang/data/synthetic_data/syn_xview_bkg_{}'.format(cmt),
                        help="Path to folder containing synthetic images and annos ")
        parser.add_argument("--syn_annos_dir", type=str, default='/data/users/yang/data/synthetic_data/syn_xview_bkg_{}_txt_xcycwh/'.format(cmt),
                            help="syn xview txt")
    else: # cmt==''
        parser.add_argument("--syn_data_dir", type=str,
                            default='/data/users/yang/data/synthetic_data/syn_xview_background_{}/',
                            help="Path to folder containing synthetic images and annos ")
        parser.add_argument("--syn_annos_dir", type=str, default='/data/users/yang/data/synthetic_data/syn_xview_background_txt_xcycwh/',
                            help="syn xview txt")


    parser.add_argument("--data_xview_dir", type=str, help="to save data files",
                        default='/data/users/yang/code/yxu-yolov3-xview/data_xview/{}_cls/')
    parser.add_argument("--data_txt_dir", type=str, help="to save txt files",
                        default='/data/users/yang/data/xView_YOLO/labels/{}/{}_cls/')
    parser.add_argument("--data_annos_dir", type=str, help="to save annos files",
                        default='/data/users/yang/data/xView_YOLO/labels/{}/{}_cls_xcycwh/')
    parser.add_argument("--data_img_dir", type=str, help="images files",
                        default='/data/users/yang/data/xView_YOLO/images/{}_{}cls/')
                        
    parser.add_argument("--syn_data_list_dir", type=str, help="to syn data list files",
                        default='/data/users/yang/code/yxu-yolov3-xview/data_xview/{}_{}_cls/{}_seed{}/')
    parser.add_argument("--min_region", type=int, default=100, help="the smallest #pixels (area) to form an object")
    parser.add_argument("--link_r", type=int, default=15,
                        help="the #pixels between two connected components to be grouped")
    parser.add_argument("--val_percent", type=float, default=0.20,
                        help="Percent to split into validation (ie .25 = val set is 25% total)")
    parser.add_argument("--tile_size", type=int, default=608, help="image size")
    parser.add_argument("--resolution", type=float, default=0.3, help="resolution of synthetic data")
    parser.add_argument("--class_num", type=int, default=1, help="Number of Total Categories")
    syn_args = parser.parse_args()
    syn_args.data_xview_dir = syn_args.data_xview_dir.format(syn_args.class_num)
    syn_args.data_txt_dir = syn_args.data_txt_dir.format(syn_args.tile_size, syn_args.class_num)
    syn_args.data_annos_dir = syn_args.data_annos_dir.format(syn_args.tile_size, syn_args.class_num)
    syn_args.data_img_dir = syn_args.data_img_dir.format(syn_args.tile_size, syn_args.class_num)
    
    return syn_args


if __name__ == '__main__':

    '''
    synthetic only
    '''

#    for ix, size_pro in enumerate([0, 0.05, 0.1, 0.15]):
####        model_cmt = 'xbw_xbkg_unif_mig21_shdw_scatter_gauss_2_7_rndsolar_dynsigma_size_bias{}_RC1_v{}'.format(size_pro, ix+70)
####        cmt = 'syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_scatter_gauss_2_7_rndsolar_dynsigma_size_bias{}_RC1_v{}_color'.format(size_pro, ix+70)
##        cmt = 'syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_fixedangle_dynsigma_size_bias{}_RC1_v{}_color'.format(size_pro, ix+120)
##        model_cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_fixedangle_dynsigma_size_bias{}_RC1_v{}'.format(size_pro, ix+120)
#        cmt = 'syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_dynsigma_size_bias{}_RC1_v{}_color'.format(size_pro, ix+140)
#        model_cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_dynsigma_size_bias{}_RC1_v{}'.format(size_pro, ix+140)
#        pxwhr = 'px15whr3'

#    for ix, size_pro in enumerate([0, 0.05, 0.1, 0.15]):
#        cmt = 'syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynsigma_size_bias{}_RC5_v{}_color'.format(size_pro, ix+40)
#        model_cmt = 'xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynsigma_size_bias{}_RC5_v{}'.format(size_pro, ix+40)
#        pxwhr = 'px23whr3'

#    for i in range(1):
#        cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias0.5_RC3_v30_color'
#        model_cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias0.5_RC3_v30'   

#        cmt = 'syn_xview_bkg_px23whr3_xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias0.5_RC2_v30_color'
#        model_cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias0.5_RC2_v30'
#        pxwhr = 'px23whr3'
#        cmt = 'syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias-0.5_RC1_v52_color'
#        model_cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias-0.5_RC1_v52'
#        pxwhr = 'px15whr3'
        
#        cmt = 'syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias-0.5_RC4_v31_color'
#        model_cmt= 'xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias-0.5_RC4_v31'
        
#        cmt = 'syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias-0.5_RC5_v22_color'
#        model_cmt = 'xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias-0.5_RC5_v22'
#        pxwhr = 'px23whr3'  


#    size_ssigma = [0, 0.03, 0.06, 0.09, 0.12]     
#    for ix, ssig in enumerate(size_ssigma):
##        cmt = 'syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_bxmuller_size_bias{}_RC1_v{}'.format(ssig, ix+1)
##        model_cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_bxmuller_size_bias{}_RC1_v{}'.format(ssig, ix+1)
##        pxwhr = 'px15whr3'
#        
##        cmt = 'syn_xview_bkg_px23whr3_xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_bxmuller_size_bias{}_RC2_v{}'.format(ssig, ix+1)
##        model_cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_bxmuller_size_bias{}_RC2_v{}'.format(ssig, ix+1)
##        pxwhr = 'px23whr3'
#        
##        cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_bxmuller_size_bias{}_RC3_v{}'.format(ssig, ix+1)
##        model_cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_bxmuller_size_bias{}_RC3_v{}'.format(ssig, ix+1)
##        pxwhr = 'px23whr3'
#        
#        cmt = 'syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_bxmuller_size_bias{}_RC4_v{}'.format(ssig, ix+1)
#        model_cmt = 'xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_bxmuller_size_bias{}_RC4_v{}'.format(ssig, ix+1)
#        pxwhr = 'px23whr3'
#        
##        cmt = 'syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_bxmuller_size_bias{}_RC5_v{}'.format(ssig, ix+1)
##        model_cmt = 'xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_bxmuller_size_bias{}_RC5_v{}'.format(ssig, ix+1)
##        pxwhr = 'px23whr3'
    
    ''' optimize color with best size '''
#    color_sigma = [5, 10, 15, 20]  # 0, 
#    for ix, ssig in enumerate(color_sigma):
##        cmt = 'syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_ssig0.03_bxmuller_color_bias{}_RC1_v{}'.format(ssig, ix+11)
##        model_cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_ssig0.03_bxmuller_color_bias{}_RC1_v{}'.format(ssig, ix+11)
#        cmt = 'syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_ssig0_color_bias{}_RC1_v{}'.format(ssig, ix+50)
#        model_cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_ssig0_color_bias{}_RC1_v{}'.format(ssig, ix+50)
#        pxwhr = 'px15whr3'
        
#        cmt = 'syn_xview_bkg_px23whr3_xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_ssig0.12_bxmuller_color_bias{}_RC2_v{}'.format(ssig, ix+11)
#        model_cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_ssig0.12_bxmuller_color_bias{}_RC2_v{}'.format(ssig, ix+11)
#        cmt = 'syn_xview_bkg_px23whr3_xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_ssig0.03_bxmuller_color_bias{}_RC2_v{}'.format(ssig, ix+31)
#        model_cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_ssig0.03_bxmuller_color_bias{}_RC2_v{}'.format(ssig, ix+31)
#        pxwhr = 'px23whr3'
        
#        cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0_bxmuller_color_bias{}_RC3_v{}'.format(ssig, ix+11)
#        model_cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0_bxmuller_color_bias{}_RC3_v{}'.format(ssig, ix+11)
#        pxwhr = 'px23whr3'
        
#        cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_bxmuller_color_bias{}_RC4_v{}'.format(ssig, ix+11)
#        model_cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_bxmuller_color_bias{}_RC4_v{}'.format(ssig, ix+11)
#        cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.06_bxmuller_color_bias{}_RC4_v{}'.format(ssig, ix+31)
#        model_cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.06_bxmuller_color_bias{}_RC4_v{}'.format(ssig, ix+31)
#        cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_square_bias{}_RC4_v{}'.format(ssig, ix+71)
#        model_cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_square_bias{}_RC4_v{}'.format(ssig, ix+71)
#        pxwhr = 'px23whr3'
        
#        cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.06_bxmuller_color_bias{}_RC5_v{}'.format(ssig, ix+11)
#        model_cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.06_bxmuller_color_bias{}_RC5_v{}'.format(ssig, ix+11)
#        cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.03_bxmuller_color_bias{}_RC5_v{}'.format(ssig, ix+31)
#        model_cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.03_bxmuller_color_bias{}_RC5_v{}'.format(ssig, ix+31)
#        pxwhr = 'px23whr3'
#        sd = 17
#        base_pxwhrs = 'px23whr3_seed{}'
#        syn_args = get_syn_args(model_cmt)
#        base_pxwhrs = base_pxwhrs.format(sd)
#        split_syn_xview_background_trn_val(sd, cmt, pxwhr, base_pxwhrs)
#        create_syn_data(cmt, sd, base_pxwhrs, val_xview=False)
    
#    cmt = 'syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_ssig0_texture_RC1_v80'
#    model_cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_ssig0_texture_RC1_v80'
#    pxwhr = 'px15whr3'
#    px_thres = 15
    
#    cmt = 'syn_xview_bkg_px23whr3_xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_ssig0_texture_RC2_v80'
#    model_cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_ssig0_texture_RC2_v80'
#    pxwhr = 'px23whr3'
#    px_thres = 23
    
#    cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_texture_RC4_v80'
#    model_cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_texture_RC4_v80'
#    px_thres = 23
#    pxwhr = 'px23whr3'
#
#    
#    sd = 17
#    base_pxwhrs = 'px23whr3_seed{}'
#    syn_args = get_syn_args(model_cmt)
#    base_pxwhrs = base_pxwhrs.format(sd)
#    split_syn_xview_background_trn_val(sd, cmt, pxwhr, base_pxwhrs, display_type = 'texture')
#    create_syn_data(cmt, sd, base_pxwhrs, val_xview=False)    

    ''' optimize color with best size '''
    color_sigma = [0]  #0, , 15, 30, 45, 60
    for ix, ssig in enumerate(color_sigma):
#        cmt = 'syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_ssig0_color_bias{}_RC1_v{}'.format(ssig, ix+51)
#        model_cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_ssig0_color_bias{}_RC1_v{}'.format(ssig, ix+51)
#        pxwhr = 'px15whr3'

#        cmt = 'syn_xview_bkg_px23whr3_xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_ssig0_color_bias{}_RC2_v{}'.format(ssig, ix+50)
#        model_cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_ssig0_color_bias{}_RC2_v{}'.format(ssig, ix+50)
#        pxwhr = 'px23whr3'

        cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_bias{}_RC3_v{}'.format(ssig, ix+50)
        model_cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_bias{}_RC3_v{}'.format(ssig, ix+50)
        pxwhr = 'px23whr3'

#        cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_bias{}_RC4_v{}'.format(ssig, ix+50)
#        model_cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_bias{}_RC4_v{}'.format(ssig, ix+50)
#        pxwhr = 'px23whr3'

#        cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.03_color_bias{}_RC5_v{}'.format(ssig, ix+50)
#        model_cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.03_color_bias{}_RC5_v{}'.format(ssig, ix+50)
#        pxwhr = 'px23whr3'
        

        sd = 17
        base_pxwhrs = 'px23whr3_seed{}'
        syn_args = get_syn_args(model_cmt)
        base_pxwhrs = base_pxwhrs.format(sd)
        split_syn_xview_background_trn_val(sd, cmt, pxwhr, base_pxwhrs)
        create_syn_data(cmt, sd, base_pxwhrs, val_xview=False)


    ''' promu bxmuller size '''
#    size_sigma = [0, 0.03, 0.06, 0.09, 0.12]   # 0]#,  
#    for ix, ssig in enumerate(size_sigma):
##        cmt = 'syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_promu_size_bias{}_RC1_v{}'.format(ssig, ix+40)
##        model_cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_promu_size_bias{}_RC1_v{}'.format(ssig, ix+40)
##        pxwhr = 'px15whr3'
#        
##        cmt = 'syn_xview_bkg_px23whr3_xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_promu_size_bias{}_RC2_v{}'.format(ssig, ix+40)
##        model_cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_promu_size_bias{}_RC2_v{}'.format(ssig, ix+40)
##        pxwhr = 'px23whr3'
#        
##        cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_size_bias{}_RC3_v{}'.format(ssig, ix+40)
##        model_cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_size_bias{}_RC3_v{}'.format(ssig, ix+40)
##        pxwhr = 'px23whr3'
#        
##        cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_size_bias{}_RC4_v{}'.format(ssig, ix+40)
##        model_cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_size_bias{}_RC4_v{}'.format(ssig, ix+40)
##        pxwhr = 'px23whr3'
#        
#        cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_size_bias{}_RC5_v{}'.format(ssig, ix+40)
#        model_cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_size_bias{}_RC5_v{}'.format(ssig, ix+40)
#        pxwhr = 'px23whr3'                
#                
#        sd = 17
#        base_pxwhrs = 'px23whr3_seed{}'
#        syn_args = get_syn_args(model_cmt)
#        base_pxwhrs = base_pxwhrs.format(sd)
#        split_syn_xview_background_trn_val(sd, cmt, pxwhr, base_pxwhrs)
#        create_syn_data(cmt, sd, base_pxwhrs, val_xview=False)

    ''' fixedsolar for best size and color '''
#    cmt = 'syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_fixedsolar_ssig0.03_csig20_RC1_v30'
#    model_cmt = 'xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_fixedsolar_ssig0.03_csig20_RC1_v30'
#    pxwhr  = 'px15whr3'

#    cmt = 'syn_xview_bkg_px23whr3_xbsw_xwing_xbkg_shdw_split_scatter_gauss_fixedsolar_ssig0.12_csig10_RC2_v30'
#    model_cmt = 'xbsw_xwing_xbkg_shdw_split_scatter_gauss_fixedsolar_ssig0.12_csig10_RC2_v30'
#    pxwhr  = 'px23whr3'
 
#    cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_fixedsolar_ssig0_csig0_RC3_v30'
#    model_cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_fixedsolar_ssig0_csig0_RC3_v30'
#    pxwhr  = 'px23whr3'
 
#    cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_fixedsolar_ssig0.09_csig0_RC4_v30'
#    model_cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_fixedsolar_ssig0.09_csig0_RC4_v30'
#    pxwhr  = 'px23whr3' 

#    cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_fixedsolar_ssig0.06_csig10_RC5_v30'
#    model_cmt = 'xbw_xbkg_unif_shdw_split_scatter_gauss_fixedsolar_ssig0.06_csig10_RC5_v30'
#    pxwhr  = 'px23whr3' 
    
#    sd = 17
#    base_pxwhrs = 'px23whr3_seed{}'
#    syn_args = get_syn_args(model_cmt)
#    base_pxwhrs = base_pxwhrs.format(sd)
#    split_syn_xview_background_trn_val(sd, cmt, pxwhr, base_pxwhrs)
#    create_syn_data(cmt, sd, base_pxwhrs, val_xview=False)
