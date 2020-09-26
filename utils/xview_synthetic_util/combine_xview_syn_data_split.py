import glob
import numpy as np
import argparse
import os
import pandas as pd
# from utils.data_process_distribution_vis_util import process_wv_coco_for_yolo_patches_no_trnval as pwv
# from utils.xview_synthetic_util import preprocess_xview_syn_data_distribution as pps
# from utils.object_score_util import get_bbox_coords_from_annos_with_object_score as gbc

def combine_xview_syn(comment='', name='xview_rc', seed=17):
    args = get_args()

    syn_data_dir = args.syn_data_dir.format(comment, args.class_num)
    syn_img_txt = pd.read_csv(open(os.path.join(syn_data_dir, '{}_seed{}'.format(comment, seed),  '{}_train_img_seed{}.txt'.format(comment, seed))), header=None).to_numpy()
    syn_lbl_txt = pd.read_csv(open(os.path.join(syn_data_dir, '{}_seed{}'.format(comment, seed), '{}_train_lbl_seed{}.txt'.format(comment, seed))), header=None).to_numpy()
    syn_trn_num = syn_img_txt.shape[0]

    base_cmt = 'px23whr3_seed{}'.format(seed)
<<<<<<< HEAD
    data_xview_dir = args.data_xview_dir
=======
    data_xview_dir = args.data_xview_dir.format(args.class_num)
>>>>>>> master
    xview_img_txt = pd.read_csv(open(os.path.join(data_xview_dir, base_cmt, '{}train_img_{}.txt'.format(name, base_cmt))), header=None).to_numpy()
    xview_lbl_txt = pd.read_csv(open(os.path.join(data_xview_dir, base_cmt, '{}train_lbl_{}.txt'.format(name, base_cmt))), header=None).to_numpy()
    xview_trn_num = xview_img_txt.shape[0]

    xview_syn_trn_txt_list_dir = os.path.join(data_xview_dir, 'xview_{}'.format(comment))
    if not os.path.exists(xview_syn_trn_txt_list_dir):
        os.mkdir(xview_syn_trn_txt_list_dir)

<<<<<<< HEAD
    f_img = open(os.path.join(xview_syn_trn_txt_list_dir, 'xview_{}_seed{}'.format(comment, seed), 'xview_{}_train_img_seed{}.txt'.format(comment, seed)), 'w')
    f_lbl = open(os.path.join(xview_syn_trn_txt_list_dir, 'xview_{}_seed{}'.format(comment, seed), 'xview_{}_train_lbl_seed{}.txt'.format(comment, seed)), 'w')
=======
    f_img = open(os.path.join(xview_syn_trn_txt_list_dir, 'xview_{}_train_img_seed{}.txt'.format(comment, seed)), 'w')
    f_lbl = open(os.path.join(xview_syn_trn_txt_list_dir, 'xview_{}_train_lbl_seed{}.txt'.format(comment, seed)), 'w')
>>>>>>> master

    for ix in range(xview_trn_num):
        f_img.write("%s\n" % xview_img_txt[ix, 0])
        f_lbl.write("%s\n" % xview_lbl_txt[ix, 0])

    for i in range(syn_trn_num):
        f_img.write("%s\n" % syn_img_txt[i, 0])
        f_lbl.write("%s\n" % syn_lbl_txt[i, 0])
    f_img.close()
    f_lbl.close()


def create_xview_syn_data(comment='', name='xview_rc', seed=17):
    args = get_args()

    syn_data_dir = args.syn_data_dir.format(comment, args.class_num)

    base_cmt = 'px23whr3_seed{}'.format(seed)
<<<<<<< HEAD
    data_xview_dir = args.data_xview_dir
=======
    data_xview_dir = args.data_xview_dir.format(args.class_num)
>>>>>>> master
    xview_img_txt = pd.read_csv(open(os.path.join(data_xview_dir, base_cmt, '{}train_img_{}.txt'.format(name, base_cmt))), header=None).to_numpy()
    xview_trn_num = xview_img_txt.shape[0]

    xview_syn_trn_txt_list_dir = os.path.join(data_xview_dir, 'xview_{}_seed{}'.format(comment, seed))
    if not os.path.exists(xview_syn_trn_txt_list_dir):
        os.mkdir(xview_syn_trn_txt_list_dir)

    data_txt = open(os.path.join(xview_syn_trn_txt_list_dir, 'xview_{}_seed{}.data'.format(comment, seed)), 'w')
    data_txt.write(
        'xview_train={}\n'.format(os.path.join(data_xview_dir, base_cmt, '{}train_img_{}.txt'.format(name, base_cmt))))
    data_txt.write(
        'xview_train_label={}\n'.format(os.path.join(data_xview_dir, base_cmt, '{}train_lbl_{}.txt'.format(name, base_cmt))))

    data_txt.write(
        'syn_train={}\n'.format(os.path.join(syn_data_dir, '{}_seed{}'.format(comment, seed),  '{}_train_img_seed{}.txt'.format(comment, seed))))
    data_txt.write(
        'syn_train_label={}\n'.format(os.path.join(syn_data_dir, '{}_seed{}'.format(comment, seed), '{}_train_lbl_seed{}.txt'.format(comment, seed))))
    data_txt.write(
        'valid={}\n'.format(os.path.join(data_xview_dir, base_cmt, '{}val_img_{}.txt'.format(name, base_cmt))))
    data_txt.write(
        'valid_label={}\n'.format(os.path.join(data_xview_dir, base_cmt, '{}val_lbl_{}.txt'.format(name, base_cmt))))
<<<<<<< HEAD
=======

>>>>>>> master
    data_txt.write('syn_0_xview_number={}\n'.format(xview_trn_num))
    data_txt.write('classes=%s\n' % str(args.class_num))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval=color\n')
    data_txt.close()



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_xview_dir", type=str, help="to save data files",
<<<<<<< HEAD
                        #default='/media/lab/Yang/code/yolov3/data_xview/{}_cls/')
                        default='/data/users/yang/code/yxu-yolov3-xview/data_xview/{}_cls/')

    parser.add_argument("--syn_data_dir", type=str, help="to syn data list files",
                        #default='/media/lab/Yang/code/yolov3/data_xview/{}_{}_cls/')
                        default='/data/users/yang/code/yxu-yolov3-xview/data_xview/{}_{}_cls/')
=======
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_cls/{}/')
                        # default='/data/users/yang/code/yxu-yolov3-xview/data_xview/{}_cls/')

    parser.add_argument("--syn_data_dir", type=str, help="to syn data list files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_{}_cls/')
                        # default='/data/users/yang/code/yxu-yolov3-xview/data_xview/{}_{}_cls/')
>>>>>>> master

    parser.add_argument("--val_percent", type=float, default=0.20,
                        help="Percent to split into validation (ie .25 = val set is 25% total)")
    parser.add_argument("--font3", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 10}")
    parser.add_argument("-ft2", "--font2", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 13}")

    parser.add_argument("--class_num", type=int, default=1, help="Number of Total Categories")  # 60  6 1
    parser.add_argument("--seed", type=int, default=17, help="random seed") #fixme -- 1024 17
    parser.add_argument("--input_size", type=int, default=608, help="image size")  # 300 416

    parser.add_argument("--min_region", type=int, default=100, help="the smallest #pixels (area) to form an object")
    parser.add_argument("--link_r", type=int, default=15,
                        help="the #pixels between two connected components to be grouped")

    parser.add_argument("--resolution", type=float, default=0.3, help="resolution of synthetic data")

<<<<<<< HEAD
    parser.add_argument("--cities", type=str,
                        default="['barcelona', 'berlin', 'francisco', 'hexagon', 'radial', 'siena', 'spiral']",
                        help="the synthetic data of cities")
    parser.add_argument("--streets", type=str, default="[200, 200, 200, 200, 200, 250, 130]",
                        help="the  #streets of synthetic  cities ")
    args = parser.parse_args()
    args.data_xview_dir = args.data_xview_dir.format(args.class_num)
=======
    args = parser.parse_args()
>>>>>>> master
    return args


if __name__ == '__main__':
<<<<<<< HEAD
    
    comments = ['syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynsigma_size_bias0.15_RC5_v43_color']
    for cmt in comments:
        create_xview_syn_data(cmt, name='xview_rc', seed=17)
=======

    comments = []
    for cmt in comments:
        create_xview_syn_data(cmt, name='xview_rc', seed=17)
>>>>>>> master
