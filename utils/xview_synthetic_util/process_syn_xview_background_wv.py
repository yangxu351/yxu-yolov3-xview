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


def split_syn_xview_background_trn_val(seed=1024, comment='syn_xview_background_texture'):

    data_dir = syn_args.syn_data_list_dir.format(comment, syn_args.class_num, seed)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    display_type = comment.split('_')[-1]
    step = syn_args.tile_size * syn_args.resolution
    all_files = np.sort(glob.glob(os.path.join(syn_args.syn_data_dir.format(display_type), '{}_all_images_step{}'.format(display_type, step), '*' + IMG_FORMAT)))
    num_files = len(all_files)
    # base_trn = np.loadtxt(os.path.join(syn_args.data_xview_dir, 'px6whr4_ng0_seed1024/xviewtrain_img_px6whr4_ng0_seed1024.txt'), dtype='str')
    # num_trn = base_trn.shape[0]
    # base_val = np.loadtxt(os.path.join(syn_args.data_xview_dir, 'px6whr4_ng0_seed1024/xviewval_img_px6whr4_ng0_seed1024.txt'), dtype='str')
    # num_val = base_val.shape[0]
    num_trn = int(num_files * (1-syn_args.val_percent))
    num_val = num_files - num_trn

    np.random.seed(seed)
    all_indices = np.random.permutation(num_files)

    trn_img_txt = open(os.path.join(data_dir, '{}_train_img_seed{}.txt'.format(comment, seed)), 'w')
    trn_lbl_txt = open(os.path.join(data_dir, '{}_train_lbl_seed{}.txt'.format(comment, seed)), 'w')

    val_img_txt = open(os.path.join(data_dir, '{}_val_img_seed{}.txt'.format(comment, seed)), 'w')
    val_lbl_txt = open(os.path.join(data_dir, '{}_val_lbl_seed{}.txt'.format(comment, seed)), 'w')

    lbl_dir = os.path.join(syn_args.syn_annos_dir, 'minr{}_linkr{}_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, display_type, step))
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


def create_syn_data(comment='syn_xview_background_texture', seed=1024):
    data_dir = syn_args.syn_data_list_dir.format(comment, syn_args.class_num, seed)
    dt = comment.split('_')[-1]
    data_txt = open(os.path.join(data_dir, '{}_seed{}.data'.format(comment, seed)), 'w')
    data_txt.write('train=./data_xview/{}_{}_cls_seed{}/{}_train_img_seed{}.txt\n'.format(comment, syn_args.class_num, seed, comment, seed))
    data_txt.write('train_label=./data_xview/{}_{}_cls_seed{}/{}_train_lbl_seed{}.txt\n'.format(comment, syn_args.class_num, seed, comment, seed))

     #fixme **********
    # df = pd.read_csv(os.path.join(data_dir, '{}_train_img_seed{}.txt'.format(comment, seed)), header=None) # **********
    data_txt.write('syn_0_xview_number=374\n')
    data_txt.write('classes=%s\n' % str(syn_args.class_num))
    data_txt.write('valid=./data_xview/{}_{}_cls_seed{}/{}_val_img_seed{}.txt\n'.format(comment, syn_args.class_num, seed, comment, seed))
    data_txt.write('valid_label=./data_xview/{}_{}_cls_seed{}/{}_val_lbl_seed{}.txt\n'.format(comment, syn_args.class_num, seed, comment, seed))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(syn_args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval={}'.format(dt))
    data_txt.close()


def split_syn_xview_background_trn_val_of_ratios(ratios, seed=1024, comment='xview_syn_xview_bkg_texture'):
    display_type = comment.split('_')[-1]
    data_dir = os.path.join(syn_args.data_xview_dir.format(syn_args.class_num), 'xview_syn_xview_bkg_{}_seed{}'.format(display_type, seed))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    step = syn_args.tile_size * syn_args.resolution

    all_files = np.sort(glob.glob(os.path.join(syn_args.syn_data_dir.format(display_type), '{}_all_images_step{}'.format(display_type, step), '*' + IMG_FORMAT)))
    num_files = len(all_files)

    base_trn_img = np.loadtxt(os.path.join(syn_args.data_xview_dir, 'px6whr4_ng0_seed{}/xviewtrain_img_px6whr4_ng0_seed{}.txt'.format(seed, seed)), dtype='str')
    base_trn_lbl = np.loadtxt(os.path.join(syn_args.data_xview_dir, 'px6whr4_ng0_seed{}/xviewtrain_lbl_px6whr4_ng0_seed{}.txt'.format(seed, seed)), dtype='str')
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

        lbl_dir = os.path.join(syn_args.syn_annos_dir, 'minr{}_linkr{}_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, display_type, step))
        for j in all_indices[:num_trn_syn]:
            trn_img_txt.write('%s\n' % all_files[j])
            trn_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(all_files[j]).replace(IMG_FORMAT, TXT_FORMAT)))
        trn_img_txt.close()
        trn_lbl_txt.close()


def create_syn_data_by_ratio(ratio, comment='xview_syn_xview_bkg_texture', seed=1024):
    dt = comment.split('_')[-1]
    data_dir = os.path.join(syn_args.data_xview_dir.format(syn_args.class_num), '{}_seed{}'.format(comment, seed))

    data_txt = open(os.path.join(data_dir, '{}_seed{}_{}xSyn.data'.format(comment, seed, ratio)), 'w')
    data_txt.write('train=./data_xview/{}_cls/{}_seed{}/{}_train_img_seed{}_{}xSyn.txt\n'.format(syn_args.class_num, comment, seed, comment, seed, r))
    data_txt.write('train_label=./data_xview/{}_cls/{}_seed{}/{}_train_lbl_seed{}_{}xSyn.txt\n'.format(syn_args.class_num, comment, seed, comment, seed, r))

     #fixme **********
    # df = pd.read_csv(os.path.join(data_dir, '{}_train_img_seed{}.txt'.format(comment, seed)), header=None) # **********
    data_txt.write('syn_0_xview_number=374\n')
    data_txt.write('classes=%s\n' % str(syn_args.class_num))
    os.path.join(syn_args.data_xview_dir, 'px6whr4_ng0_seed{}/xviewval_img_px6whr4_ng0_seed{}.txt'.format(seed, seed))
    data_txt.write('valid=./data_xview/{}_cls/px6whr4_ng0_seed{}/xviewval_img_px6whr4_ng0_seed{}.txt\n'.format(syn_args.class_num, seed, seed))
    data_txt.write('valid_label=./data_xview/{}_cls/px6whr4_ng0_seed{}/xviewval_lbl_px6whr4_ng0_seed{}.txt\n'.format(syn_args.class_num, seed, seed))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(syn_args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval={}'.format(dt))
    data_txt.close()


def get_syn_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--syn_data_dir", type=str,
                        default='/media/lab/Yang/data/synthetic_data/syn_xview_background_{}/',
                        help="Path to folder containing synthetic images and annos ")
    parser.add_argument("--syn_annos_dir", type=str, default='/media/lab/Yang/data/synthetic_data/syn_xview_background_txt_xcycwh',
                        help="syn xview txt")

    parser.add_argument("--data_xview_dir", type=str, help="to save data files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_cls/')
    parser.add_argument("--syn_data_list_dir", type=str, help="to syn data list files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_{}_cls_seed{}/')
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
    syn_args = get_syn_args()

    # comments = ['syn_xview_background_texture', 'syn_xview_background_color', 'syn_xview_background_mixed']
    # seeds = [1024, 17, 3, 5, 9]
    # for cmt in comments:
    #     for sd in seeds:
    #         split_syn_xview_background_trn_val(sd, cmt)

    # comments = ['syn_xview_background_texture', 'syn_xview_background_color', 'syn_xview_background_mixed']
    # seeds = [1024, 17, 3, 5, 9]
    # for cmt in comments:
    #     for sd in seeds:
    #         create_syn_data(cmt, sd)

    # comments =['xview_syn_xview_bkg_texture', 'xview_syn_xview_bkg_color', 'xview_syn_xview_bkg_mixed']
    # seeds = [1024, 17, 3, 5, 9]
    # syn_ratios = [1, 2]
    # for cmt in comments:
    #     for sd in seeds:
    #         split_syn_xview_background_trn_val_of_ratios(syn_ratios, sd, cmt)

    comments = ['xview_syn_xview_bkg_texture', 'xview_syn_xview_bkg_color', 'xview_syn_xview_bkg_mixed']
    seeds = [1024, 17, 3, 5, 9]
    syn_ratios = [1, 2]
    for cmt in comments:
        for sd in seeds:
            for r in syn_ratios:
                create_syn_data_by_ratio(r, cmt, sd)
