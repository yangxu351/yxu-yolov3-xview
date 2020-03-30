import glob
import numpy as np
import argparse
import os
import pandas as pd
from ast import literal_eval
import seaborn as sns
from matplotlib import pyplot as plt
import json
import shutil
import cv2

IMG_FORMAT0 = '.jpg'
IMG_FORMAT = '.png'
TXT_FORMAT = '.txt'


def get_part_syn_args(dt, sr):
    parser = argparse.ArgumentParser()

    parser.add_argument("--syn_bkg_img_anno_dir", type=str, help="images and annotations of synthetic airplanes",
                        default='/media/lab/Yang/data/synthetic_data/syn_background/')

    parser.add_argument("--syn_bkg_txt_dir", type=str, help="txt labels of synthetic airplanes",
                        default='/media/lab/Yang/data/synthetic_data/syn_background_txt_xcycwh/{}/')

    parser.add_argument("--syn_bkg_gt_bbox_dir", type=str, help="gt images with bbox of synthetic airplanes",
                        default='/media/lab/Yang/data/synthetic_data/syn_background_gt_bbox/{}/')

    parser.add_argument("--syn_images_save_dir", type=str, help="rgb images of synthetic airplanes",
                        default='/media/lab/Yang/data/xView_YOLO/images/{}_{}/')
    parser.add_argument("--syn_annos_save_dir", type=str, help="gt of synthetic airplanes",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_{}_cls_xcycwh/')
    parser.add_argument("--syn_txt_save_dir", type=str, help="gt related files of synthetic airplanes",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_{}_cls/')

    parser.add_argument("--data_txt_dir", type=str, help="to save txt files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_cls/')

    parser.add_argument("--data_xview_dir", type=str, help="to save data files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_cls/')

    parser.add_argument("--syn_data_list_dir", type=str, help="to syn data list files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_{}_cls/')

    parser.add_argument("--results_dir", type=str, help="to save category files",
                        default='/media/lab/Yang/code/yolov3/result_output/{}_cls/{}_{}/')

    parser.add_argument("--cat_sample_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/{}/{}_cls/')

    parser.add_argument("--val_percent", type=float, default=0.20,
                        help="Percent to split into validation (ie .25 = val set is 25% total)")
    parser.add_argument("--font3", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 10}")
    parser.add_argument("-ft2", "--font2", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 13}")

    parser.add_argument("--class_num", type=int, default=1, help="Number of Total Categories")  # 60  6
    parser.add_argument("--tile_size", type=int, default=608, help="image size")  # 300 416

    parser.add_argument("--syn_display_type", type=str, default=dt,
                        help="syn_background")  # ######*********************change

    parser.add_argument("--min_region", type=int, default=100, help="the smallest #pixels (area) to form an object")
    parser.add_argument("--link_r", type=int, default=15,
                        help="the #pixels between two connected components to be grouped")

    parser.add_argument("--resolution", type=float, default=0.3, help="resolution of synthetic data")

    parser.add_argument("--cities", type=str,
                        default="['barcelona', 'berlin', 'francisco', 'hexagon', 'radial', 'siena', 'spiral']",
                        help="the synthetic data of cities")
    parser.add_argument("--streets", type=str, default="[200, 200, 200, 200, 200, 250, 130]",
                        help="the  #streets of synthetic  cities ")
    syn_args = parser.parse_args()
    syn_args.cities = literal_eval(syn_args.cities)
    syn_args.streets = literal_eval(syn_args.streets)
    syn_args.data_xview_dir = syn_args.data_xview_dir.format(syn_args.class_num)
    syn_args.data_txt_dir = syn_args.data_txt_dir.format(syn_args.tile_size, syn_args.class_num)
    syn_args.cat_sample_dir = syn_args.cat_sample_dir.format(syn_args.tile_size, syn_args.class_num)

    syn_args.syn_images_save_dir = syn_args.syn_images_save_dir.format(syn_args.tile_size, syn_args.syn_display_type)
    syn_args.syn_annos_save_dir = syn_args.syn_annos_save_dir.format(syn_args.tile_size, syn_args.syn_display_type, syn_args.class_num)
    syn_args.syn_txt_save_dir = syn_args.syn_txt_save_dir.format(syn_args.tile_size, syn_args.syn_display_type, syn_args.class_num)

    return syn_args


if __name__ == '__main__':
    # ('%10s' * 2 + '%10.3g' * 6) % (
    #                 '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, len(targets), img_size)
    # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

    map_comp = np.zeros((4, 2))
    dt = 'syn_background'
    comments = '_px6whr4_ng0_seed1024'
    syn_ratio = [0, 0.1, 0.2, 0.3]
    step = 171
    for ix, sr in enumerate(syn_ratio):
        syn_args = get_part_syn_args(dt, sr)
        syn_args.results_dir = syn_args.results_dir.format(syn_args.class_num, syn_args.syn_display_type, sr)
        result_file = np.sort(glob.glob(os.path.join(syn_args.results_dir, '*' + comments.split('_seed')[0], 'results_{}_{}.txt'.format(syn_args.syn_display_type, sr))))[-1]
        map_comp[ix, 0] = np.loadtxt(result_file, usecols=[10])[step] # 0, 1,  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    comments1 = '_px6whr4_ng0_seed17'
    for jx, sr in enumerate(syn_ratio):
        syn_args = get_part_syn_args(dt, sr)
        syn_args.results_dir = syn_args.results_dir.format(syn_args.class_num, syn_args.syn_display_type, sr)
        result_file = np.sort(glob.glob(os.path.join(syn_args.results_dir, '*' + comments1, 'results_{}_{}.txt'.format(syn_args.syn_display_type, sr))))[-1]
        map_comp[jx, 1] = np.loadtxt(result_file, usecols=[10])[step] # 0, 1,  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    comments2 = '_px6whr4_ng0_seed1024_mosaic_rect'
    for ix, sr in enumerate(syn_ratio):
        syn_args = get_part_syn_args(dt, sr)
        syn_args.results_dir = syn_args.results_dir.format(syn_args.class_num, syn_args.syn_display_type, sr)
        result_file = np.sort(glob.glob(os.path.join(syn_args.results_dir, '*' + comments2.split('_seed')[0], 'results_{}_{}.txt'.format(syn_args.syn_display_type, sr))))[-1]
        map_comp[ix, 0] = np.loadtxt(result_file, usecols=[10])[step] # 0, 1,  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    comments3 = '_px6whr4_ng0_seed17_mosaic_rect'
    for jx, sr in enumerate(syn_ratio):
        syn_args = get_part_syn_args(dt, sr)
        syn_args.results_dir = syn_args.results_dir.format(syn_args.class_num, syn_args.syn_display_type, sr)
        result_file = np.sort(glob.glob(os.path.join(syn_args.results_dir, '*' + comments3, 'results_{}_{}.txt'.format(syn_args.syn_display_type, sr))))[-1]
        map_comp[jx, 1] = np.loadtxt(result_file, usecols=[10])[step] # 0, 1,  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    save_dir = os.path.join(syn_args.data_txt_dir, 'MAP_comp', dt)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df_map = pd.DataFrame(map_comp, columns=[comments[1:], comments1[1:]], index=syn_ratio)
    df_map.to_csv(os.path.join(save_dir, '{}{}_VS_seed1024.csv'.format(dt, comments1)))
    '''
    syn_background_with_different_ratio (0, 0.1, 0.2, 0.3)
    '''
    boxplot = df_map.boxplot(column=[df_map.columns[0], df_map.columns[1]], return_type='axes')
    boxplot.set_title('syn_background_with_different_ratio (0, 0.1, 0.2, 0.3)')
    boxplot.set_xlabel('{}_seed'.format(dt))
    boxplot.set_ylabel('MAP')
    boxplot.figure.savefig(os.path.join(save_dir, '{}_different_ratio_boxplot.jpg'.format(dt, comments1)))
    '''
    syn_background_with_different_seed (1024, 17)
    '''
    # df_map = df_map.T
    # boxplot = df_map.boxplot(column=[df_map.columns[0], df_map.columns[1], df_map.columns[2], df_map.columns[3]], return_type='axes')
    # boxplot.set_title('{}_with_different_seed (1024, 17)'.format(dt))
    # boxplot.set_xlabel('{}_ratio'.format(dt))
    # boxplot.set_ylabel('MAP')
    # boxplot.figure.savefig(os.path.join(save_dir, '{}_different_seed_boxplot.jpg'.format(dt, comments1)))

    # fig, ax = plt.subplots()
    # sns.barplot(x=df_map.columns[0], y=df_map.columns[1], data=df_map,  ax=ax)
    # ax2 = ax.twinx()
    # sns.barplot(x=df_map.columns[0], y=df_map.columns[2], data=df_map,  ax=ax2)

    # sns.barplot(x=df_map.columns[0], y=df_map.columns[1], hue=df_map.columns[1], data=df_map, ax=ax, ci='sd', saturation=0.7)
    # ax2 = ax.twinx()
    # sns.barplot(x=df_map.columns[0], y=df_map.columns[2], hue=df_map.columns[2], data=df_map, ax=ax2, ci='sd', saturation=0.7)

    # tips = sns.load_dataset("tips")
    # df_map.plot.bar(x='syn_background_ratio', rot=0)
    # df_map.boxplot()
    # plt.boxplot(x=df_map.loc[:, 0], y=)
