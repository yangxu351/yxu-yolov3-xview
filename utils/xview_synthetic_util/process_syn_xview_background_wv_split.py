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


def split_syn_xview_background_trn_val(seed=17, comment='syn_xview_background_texture', pxwhrs=''):

    data_xview_dir = syn_args.data_xview_dir.format( syn_args.class_num)
    df_trn = pd.read_csv(os.path.join(data_xview_dir,pxwhrs, 'xviewtrain_img_{}.txt'.format(pxwhrs)), header=None)
    num_trn = df_trn.shape[0]
    df_val = pd.read_csv(os.path.join(data_xview_dir, pxwhrs, 'xviewval_img_{}.txt'.format(pxwhrs)), header=None)
    num_val = df_val.shape[0]

    display_type = comment.split('_')[-1]
    step = syn_args.tile_size * syn_args.resolution
    all_files = np.sort(glob.glob(os.path.join(syn_args.syn_data_dir.format(display_type), '{}_all_images_step{}'.format(display_type, step), '*' + IMG_FORMAT)))
    num_files = len(all_files)

    np.random.seed(seed)
    all_indices = np.random.permutation(num_files)
    data_dir = syn_args.syn_data_list_dir.format( comment, syn_args.class_num, comment, seed)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    trn_img_txt = open(os.path.join(data_dir, '{}_train_img_seed{}.txt'.format(comment, seed)), 'w')
    trn_lbl_txt = open(os.path.join(data_dir, '{}_train_lbl_seed{}.txt'.format(comment, seed)), 'w')

    val_img_txt = open(os.path.join(data_dir, '{}_val_img_seed{}.txt'.format(comment, seed)), 'w')
    val_lbl_txt = open(os.path.join(data_dir, '{}_val_lbl_seed{}.txt'.format(comment, seed)), 'w')

    lbl_dir = os.path.join(syn_args.syn_annos_dir, 'minr{}_linkr{}_{}_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, pxwhrs.split('_seed')[0], display_type, step))
    for i in all_indices[:num_val]:
        val_img_txt.write('%s\n' % all_files[i])
        val_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(all_files[i]).replace(IMG_FORMAT, TXT_FORMAT)))
    val_img_txt.close()
    val_lbl_txt.close()
    for j in all_indices[num_val : num_val+num_trn]:
        trn_img_txt.write('%s\n' % all_files[j])
        trn_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(all_files[j]).replace(IMG_FORMAT, TXT_FORMAT)))
    trn_img_txt.close()
    trn_lbl_txt.close()


def create_syn_data(comment='syn_xview_background_texture', seed=1024, pxwhrs='', val_xview=False):
    data_dir = syn_args.syn_data_list_dir.format( comment, syn_args.class_num, comment, seed)
    dt = comment.split('_')[-1]
    if val_xview:
        data_txt = open(os.path.join(data_dir, '{}_seed{}_xview_val.data'.format(comment, seed)), 'w')
    else:
        data_txt = open(os.path.join(data_dir, '{}_seed{}.data'.format(comment, seed)), 'w')
    data_txt.write('train=./data_xview/{}_{}_cls/{}_seed{}/{}_train_img_seed{}.txt\n'.format( comment, syn_args.class_num, comment, seed, comment, seed))
    data_txt.write('train_label=./data_xview/{}_{}_cls/{}_seed{}/{}_train_lbl_seed{}.txt\n'.format(comment, syn_args.class_num, comment, seed, comment, seed))

     #fixme **********
    data_xview_dir = syn_args.data_xview_dir.format( syn_args.class_num)
    if pxwhrs:
        df = pd.read_csv(os.path.join(data_xview_dir, pxwhrs, 'xviewtrain_img_{}.txt'.format(pxwhrs)), header=None)
    else:
        df = pd.read_csv(os.path.join(data_xview_dir, '{}_train_img_seed{}.txt'.format(comment, seed)), header=None) # **********
    data_txt.write('syn_0_xview_number={}\n'.format(df.shape[0]))
    data_txt.write('classes=%s\n' % str(syn_args.class_num))
    if val_xview:
        data_txt.write('valid=./data_xview/{}_cls/{}/xviewval_img_{}.txt\n'.format(syn_args.class_num, pxwhrs, pxwhrs))
        data_txt.write('valid_label=./data_xview/{}_cls/{}/xviewval_lbl_{}.txt\n'.format(syn_args.class_num, pxwhrs, pxwhrs))
    else:
        data_txt.write('valid=./data_xview/{}_{}_cls/{}_seed{}/{}_val_img_seed{}.txt\n'.format(comment, syn_args.class_num, comment, seed, comment, seed))
        data_txt.write('valid_label=./data_xview/{}_{}_cls/{}_seed{}/{}_val_lbl_seed{}.txt\n'.format(comment, syn_args.class_num, comment, seed, comment, seed))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(syn_args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval={}'.format(dt))
    data_txt.close()


def create_syn_data_with_model(comment='syn_xview_background_texture', seed=1024, base_cmt=''):
    data_dir = syn_args.syn_data_list_dir.format( comment, syn_args.class_num, comment, seed)
    dt = comment.split('_')[-1]
    data_txt = open(os.path.join(data_dir, '{}_seed{}_data_with_model.data'.format(comment, seed)), 'w')
    data_txt.write('classes=%s\n' % str(syn_args.class_num))
    data_txt.write('valid=./data_xview/{}_cls/{}_seed{}/xviewval_img_{}_seed{}.txt\n'.format(syn_args.class_num, base_cmt, seed, base_cmt, seed))
    data_txt.write('valid_label=./data_xview/{}_cls/{}_seed{}/xviewval_lbl_{}_seed{}_with_model.txt\n'.format(syn_args.class_num, base_cmt, seed, base_cmt, seed))
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

    all_files = np.sort(glob.glob(os.path.join(syn_args.syn_data_dir.format(display_type), '{}_all_images_step{}'.format(display_type, step), '*' + IMG_FORMAT)))
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


def create_syn_data_by_ratio(ratio, comment='xview_syn_xview_bkg_texture', seed=1024, base_cmt='px6whr4_ng0'):
    dt = comment.split('_')[-1]
    data_dir = os.path.join(syn_args.data_xview_dir.format(syn_args.class_num), '{}_seed{}'.format(comment, seed))

    data_txt = open(os.path.join(data_dir, '{}_seed{}_{}xSyn.data'.format(comment, seed, ratio)), 'w')
    data_txt.write('train=./data_xview/{}_cls/{}_seed{}/{}_train_img_seed{}_{}xSyn.txt\n'.format(syn_args.class_num, comment, seed, comment, seed, ratio))
    data_txt.write('train_label=./data_xview/{}_cls/{}_seed{}/{}_train_lbl_seed{}_{}xSyn.txt\n'.format(syn_args.class_num, comment, seed, comment, seed, ratio))

     #fixme **********
    xview_dir = os.path.join(syn_args.data_xview_dir.format(syn_args.class_num), '{}_seed{}'.format(base_cmt, seed))
    df = pd.read_csv(os.path.join(xview_dir, 'xviewtrain_img_{}_seed{}.txt'.format(base_cmt, seed)), header=None) # **********
    data_txt.write('syn_0_xview_number={}\n'.format(df.shape[0]))
    data_txt.write('classes=%s\n' % str(syn_args.class_num))
    os.path.join(syn_args.data_xview_dir, '{}_seed{}/xviewval_img_{}_seed{}.txt'.format(base_cmt, seed, base_cmt, seed))
    data_txt.write('valid=./data_xview/{}_cls/{}_seed{}/xviewval_img_{}_seed{}.txt\n'.format(syn_args.class_num, base_cmt, seed, base_cmt, seed))
    data_txt.write('valid_label=./data_xview/{}_cls/{}_seed{}/xviewval_lbl_{}_seed{}.txt\n'.format(syn_args.class_num, base_cmt, seed, base_cmt, seed))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(syn_args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval={}'.format(dt))
    data_txt.close()


def create_syn_data_by_ratio_with_model(ratio, comment='xview_syn_xview_bkg_texture', seed=1024, base_cmt='px6whr4_ng0'):
    dt = comment.split('_')[-1]
    data_dir = os.path.join(syn_args.data_xview_dir.format(syn_args.class_num), '{}_seed{}'.format(comment, seed))

    data_txt = open(os.path.join(data_dir, '{}_seed{}_{}xSyn_with_model.data'.format(comment, seed, ratio)), 'w')
    data_txt.write('classes=%s\n' % str(syn_args.class_num))
    os.path.join(syn_args.data_xview_dir, '{}_seed{}/xviewval_img_{}_seed{}.txt'.format(base_cmt, seed, base_cmt, seed))
    data_txt.write('valid=./data_xview/{}_cls/{}_seed{}/xviewval_img_{}_seed{}.txt\n'.format(syn_args.class_num, base_cmt, seed, base_cmt, seed))
    data_txt.write('valid_label=./data_xview/{}_cls/{}_seed{}/xviewval_lbl_{}_seed{}_with_model.txt\n'.format(syn_args.class_num, base_cmt, seed, base_cmt, seed))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(syn_args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval={}'.format(dt))
    data_txt.close()


def get_syn_args(cmt='certain_models'):
    parser = argparse.ArgumentParser()

    if cmt:
        parser.add_argument("--syn_data_dir", type=str,
                        default='/media/lab/Yang/data/synthetic_data/syn_xview_bkg_{}'.format(cmt) + '_{}/',
                        help="Path to folder containing synthetic images and annos ")
        parser.add_argument("--syn_annos_dir", type=str, default='/media/lab/Yang/data/synthetic_data/syn_xview_bkg_{}_txt_xcycwh/'.format(cmt),
                            help="syn xview txt")
    else: # cmt==''
        parser.add_argument("--syn_data_dir", type=str,
                            default='/media/lab/Yang/data/synthetic_data/syn_xview_background_{}/',
                            help="Path to folder containing synthetic images and annos ")
        parser.add_argument("--syn_annos_dir", type=str, default='/media/lab/Yang/data/synthetic_data/syn_xview_background_txt_xcycwh/',
                            help="syn xview txt")

    parser.add_argument("--data_xview_dir", type=str, help="to save data files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_cls/')
    parser.add_argument("--syn_data_list_dir", type=str, help="to syn data list files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_{}_cls/{}_seed{}/')
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
    return syn_args


if __name__ == '__main__':

    '''
    synthetic only
    '''
    # # # comments = ['syn_xview_background_texture', 'syn_xview_background_color', 'syn_xview_background_mixed']
    # # comments = ['syn_xview_bkg_certain_models_texture', 'syn_xview_bkg_certain_models_color', 'syn_xview_bkg_certain_models_mixed']
    # # # pxwhr = 'px6whr4_ng0'
    # # pxwhr = 'px6whr4'
    # # comments = ['syn_xview_bkg_px20whr4_certain_models_texture', 'syn_xview_bkg_px20whr4_certain_models_color', 'syn_xview_bkg_px20whr4_certain_models_mixed']
    # # pxwhr = 'px20whr4'
    # # seeds = [17, 1024, 5, 9, 3]
    # # comments = ['syn_xview_bkg_px23whr4_scale_models_texture', 'syn_xview_bkg_px23whr4_scale_models_color', 'syn_xview_bkg_px23whr4_scale_models_mixed']
    # # pxwhr = 'px23whr4'
    # # seeds = [17, 5, 9] #, 1024, 3, 5, 9
    # # model_cmt = 'scale_models'
    # comments = ['syn_xview_bkg_px23whr4_small_models_color', 'syn_xview_bkg_px23whr4_small_models_mixed']
    # comments = ['syn_xview_bkg_px23whr4_small_fw_models_color', 'syn_xview_bkg_px23whr4_small_fw_models_mixed']
    # pxwhr = 'px23whr4'
    # model_cmt = 'small_models'
    # model_cmt = 'small_fw_models'

    # comments = ['syn_xview_bkg_px23whr3_6groups_models_color', 'syn_xview_bkg_px23whr3_6groups_models_mixed']
    # pxwhrs = 'px23whr3_seed{}'
    # model_cmt = '6groups_models'
    # syn_args = get_syn_args(model_cmt)
    # seeds = [17]
    # for cmt in comments:
    #     for sd in seeds:
    #         pxwhrs = pxwhrs.format(sd)
    #         split_syn_xview_background_trn_val(sd, cmt, pxwhrs)

    # # # comments = ['syn_xview_background_texture', 'syn_xview_background_color', 'syn_xview_background_mixed']
    # # comments = ['syn_xview_bkg_certain_models_texture', 'syn_xview_bkg_certain_models_color', 'syn_xview_bkg_certain_models_mixed']
    # # seeds = [17]
    # # comments = ['syn_xview_bkg_px20whr4_certain_models_texture', 'syn_xview_bkg_px20whr4_certain_models_color', 'syn_xview_bkg_px20whr4_certain_models_mixed']
    # # seeds = [1024, 17, 3, 5, 9]
    # # comments = ['syn_xview_bkg_px23whr4_scale_models_texture', 'syn_xview_bkg_px23whr4_scale_models_color', 'syn_xview_bkg_px23whr4_scale_models_mixed']
    # # model_cmt = 'scale_models'
    # # seeds = [17, 5, 9]
    # # comments = ['syn_xview_bkg_px23whr4_small_models_color', 'syn_xview_bkg_px23whr4_small_models_mixed']
    # comments = ['syn_xview_bkg_px23whr4_small_fw_models_color', 'syn_xview_bkg_px23whr4_small_fw_models_mixed']
    # pxwhr = 'px23whr4'
    comments = ['syn_xview_bkg_px23whr3_6groups_models_color', 'syn_xview_bkg_px23whr3_6groups_models_mixed']
    pxwhrs = 'px23whr3_seed{}'
    seeds = [17]
    # model_cmt = 'small_models'
    model_cmt = '6groups_models'
    syn_args = get_syn_args(model_cmt)
    for cmt in comments:
        for sd in seeds:
            pxwhrs = pxwhrs.format(sd)
            create_syn_data(cmt, sd, pxwhrs, val_xview=False)
            create_syn_data(cmt, sd, pxwhrs, val_xview=True)

    # # comments = ['syn_xview_bkg_px23whr4_scale_models_texture', 'syn_xview_bkg_px23whr4_scale_models_color', 'syn_xview_bkg_px23whr4_scale_models_mixed']
    # # seeds = [17, 5, 9]
    # # model_cmt = 'scale_models'
    # # comments = ['syn_xview_bkg_px23whr4_small_models_color', 'syn_xview_bkg_px23whr4_small_models_mixed']
    # # seeds = [17]
    # # model_cmt = 'small_models'
    # comments = ['syn_xview_bkg_px23whr4_small_fw_models_color', 'syn_xview_bkg_px23whr4_small_fw_models_mixed']
    # seeds = [17]
    # model_cmt = 'small_fw_models'
    # syn_args = get_syn_args(model_cmt)
    # base_cmt = 'px23whr4'
    # for cmt in comments:
    #     for sd in seeds:
    #         create_syn_data_with_model(cmt, sd, base_cmt)

    '''
    combine xview and synthetic 
    '''
    # # comments =['xview_syn_xview_bkg_texture', 'xview_syn_xview_bkg_color', 'xview_syn_xview_bkg_mixed']
    # # base_cmt='px6whr4_ng0'
    # # comments = ['xview_syn_xview_bkg_certain_models_texture', 'xview_syn_xview_bkg_certain_models_color', 'xview_syn_xview_bkg_certain_models_mixed']
    # # base_cmt = 'px20whr4'
    # # comments = ['xview_syn_xview_bkg_px20whr4_certain_models_texture', 'xview_syn_xview_bkg_px20whr4_certain_models_color', 'xview_syn_xview_bkg_px20whr4_certain_models_mixed']
    # # seeds = [17, 1024, 3, 5, 9]
    # # model_cmt = 'certain_models'
    # # model_cmt = 'scale_models'
    # # comments = ['xview_syn_xview_bkg_px23whr4_scale_models_texture', 'xview_syn_xview_bkg_px23whr4_scale_models_color', 'xview_syn_xview_bkg_px23whr4_scale_models_mixed']
    # # base_cmt = 'px23whr4'
    # # seeds = [17, 5, 9]
    # # comments = ['xview_syn_xview_bkg_px23whr4_small_models_color', 'xview_syn_xview_bkg_px23whr4_small_models_mixed']
    # # seeds = [17]
    # # model_cmt = 'small_models'
    # # comments = ['xview_syn_xview_bkg_px23whr4_small_fw_models_color', 'xview_syn_xview_bkg_px23whr4_small_fw_models_mixed']
    # # seeds = [17]
    # # model_cmt = 'small_fw_models'
    # # base_cmt = 'px23whr4'
    # # comments = ['xview_syn_xview_bkg_px23whr3_6groups_models_color', 'xview_syn_xview_bkg_px23whr3_6groups_models_mixed']
    # # seeds = [17]
    # # model_cmt = '6groups_models'
    # comments = ['xview_syn_xview_bkg_px23whr3_6groups_models_color', 'xview_syn_xview_bkg_px23whr3_6groups_models_mixed']
    # seeds = [17]
    # model_cmt = '6groups_models'
    # # base_cmt = 'px23whr3'
    # # comments = ['xview_syn_xview_bkg_px23whr3_small_models_color', 'xview_syn_xview_bkg_px23whr3_small_models_mixed']
    # # seeds = [17]
    # # model_cmt = 'small_models'
    # base_cmt = 'px23whr3'
    # syn_args = get_syn_args(model_cmt)
    # syn_ratios = [1] # [1, 2]
    # for cmt in comments:
    #     for sd in seeds:
    #         split_syn_xview_background_trn_val_of_ratios(syn_ratios, sd, cmt, base_cmt)

    # # comments = ['xview_syn_xview_bkg_texture', 'xview_syn_xview_bkg_color', 'xview_syn_xview_bkg_mixed']
    # # comments = ['xview_syn_xview_bkg_certain_models_texture', 'xview_syn_xview_bkg_certain_models_color', 'xview_syn_xview_bkg_certain_models_mixed']
    # # seeds = [1024, 17, 3, 5, 9]
    # # syn_ratios = [1, 2]
    # # base_cmt='px20whr4'
    # # comments = ['xview_syn_xview_bkg_px20whr4_certain_models_texture', 'xview_syn_xview_bkg_px20whr4_certain_models_color', 'xview_syn_xview_bkg_px20whr4_certain_models_mixed']
    # # seeds = [17, 1024, 3, 5, 9]
    # # model_cmt = 'certain_models'
    # # model_cmt = 'scale_models'
    # # comments = ['xview_syn_xview_bkg_px23whr4_scale_models_texture', 'xview_syn_xview_bkg_px23whr4_scale_models_color', 'xview_syn_xview_bkg_px23whr4_scale_models_mixed']
    # # base_cmt = 'px23whr4'
    # # seeds = [17, 5, 9]
    # # comments = ['xview_syn_xview_bkg_px23whr4_small_models_color', 'xview_syn_xview_bkg_px23whr4_small_models_mixed']
    # # seeds = [17]
    # # model_cmt = 'small_models'
    # # base_cmt = 'px23whr4'
    # # comments = ['xview_syn_xview_bkg_px23whr4_small_fw_models_color', 'xview_syn_xview_bkg_px23whr4_small_fw_models_mixed']
    # # seeds = [17]
    # # model_cmt = 'small_fw_models'
    # # base_cmt = 'px23whr4'
    # # comments = ['xview_syn_xview_bkg_px23whr3_6groups_models_color', 'xview_syn_xview_bkg_px23whr3_6groups_models_mixed']
    # # seeds = [17]
    # # model_cmt = '6groups_models'
    # base_cmt = 'px23whr3'
    # comments = ['xview_syn_xview_bkg_px23whr3_6groups_models_color', 'xview_syn_xview_bkg_px23whr3_6groups_models_mixed']
    # seeds = [17]
    # model_cmt = '6groups_models'
    # base_cmt = 'px23whr3'
    # # comments = ['xview_syn_xview_bkg_px23whr3_small_models_color', 'xview_syn_xview_bkg_px23whr3_small_models_mixed']
    # # seeds = [17]
    # # model_cmt = 'small_models'
    # # base_cmt = 'px23whr3'
    # syn_args = get_syn_args(model_cmt)
    # syn_ratios = [1]
    # for cmt in comments:
    #     for sd in seeds:
    #         for r in syn_ratios:
    #             create_syn_data_by_ratio(r, cmt, sd, base_cmt)

    # # model_cmt = 'scale_models'
    # # syn_args = get_syn_args(model_cmt)
    # # comments = ['xview_syn_xview_bkg_px23whr4_scale_models_texture', 'xview_syn_xview_bkg_px23whr4_scale_models_color', 'xview_syn_xview_bkg_px23whr4_scale_models_mixed']
    # # base_cmt = 'px23whr4'
    # # seeds = [17, 5, 9]
    # # comments = ['xview_syn_xview_bkg_px23whr4_small_models_color', 'xview_syn_xview_bkg_px23whr4_small_models_mixed']
    # # seeds = [17]
    # # model_cmt = 'small_models'
    # # base_cmt = 'px23whr4'
    # # comments = ['xview_syn_xview_bkg_px23whr4_small_fw_models_color', 'xview_syn_xview_bkg_px23whr4_small_fw_models_mixed']
    # # seeds = [17]
    # # model_cmt = 'small_fw_models'
    # # base_cmt = 'px23whr4'
    # # comments = ['xview_syn_xview_bkg_px23whr3_6groups_models_color', 'xview_syn_xview_bkg_px23whr3_6groups_models_mixed']
    # # seeds = [17]
    # # model_cmt = '6groups_models'
    # # base_cmt = 'px23whr3'
    # comments = ['xview_syn_xview_bkg_px23whr3_6groups_models_color', 'xview_syn_xview_bkg_px23whr3_6groups_models_mixed']
    # seeds = [17]
    # model_cmt = '6groups_models'
    # base_cmt = 'px23whr3'
    # # comments = ['xview_syn_xview_bkg_px23whr3_small_models_color', 'xview_syn_xview_bkg_px23whr3_small_models_mixed']
    # # seeds = [17]
    # # model_cmt = 'small_models'
    # # base_cmt = 'px23whr3'
    # syn_args = get_syn_args(model_cmt)
    # syn_ratios = [1]
    # for cmt in comments:
    #     for sd in seeds:
    #         for r in syn_ratios:
    #             create_syn_data_by_ratio_with_model(r, cmt, sd, base_cmt)

