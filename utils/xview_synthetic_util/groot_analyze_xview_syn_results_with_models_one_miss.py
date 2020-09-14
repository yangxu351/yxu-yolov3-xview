import glob
import numpy as np
import argparse
import os
from skimage import io, color
import pandas as pd
from ast import literal_eval
from matplotlib import pyplot as plt
import json
import shutil
import cv2
import sys
sys.path.append('/data/users/yang/code/yxu-yolov3-xview/')
from utils.parse_config_xview import parse_data_cfg
from utils.xview_synthetic_util import preprocess_xview_syn_data_distribution as pps

from utils.data_process_distribution_vis_util import process_wv_coco_for_yolo_patches_no_trnval as pwv
from utils.utils_xview import coord_iou
# from utils.xview_synthetic_util import preprocess_synthetic_data_distribution as pps
# from utils.xview_synthetic_util import anaylze_xview_syn_results as axs
IMG_SUFFIX = '.jpg'
# IMG_SUFFIX = '.png'
TXT_SUFFIX = '.txt'


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def autolabel(ax, rects, x, labels, ylabel=None, rotation=90, txt_rotation=45):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height) if height != 0 else '',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=txt_rotation)
        if rotation == 0:
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
        else:  # rotation=90, multiline label
            xticks = []
            for i in range(len(labels)):
                xticks.append('{} {}'.format(x[i], labels[i]))
            ax.set_xticklabels(xticks, rotation=rotation)
        # ax.set_ylabel(ylabel)
        # ax.grid(True)


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0




def get_part_syn_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_filepath", type=str, help="Filepath to GEOJSON coordinate file",
                        default='/data/users/yang/data/xView/xView_train.geojson')

    # parser.add_argument("--syn_plane_img_anno_dir", type=str, help="images and annotations of synthetic airplanes",
    #                     default='/data/users/yang/data/synthetic_data/Airplanes/{}/')
    #
    # parser.add_argument("--syn_plane_txt_dir", type=str, help="txt labels of synthetic airplanes",
    #                     default='/data/users/yang/data/synthetic_data/Airplanes_txt_xcycwh/{}/')
    #
    # parser.add_argument("--syn_plane_gt_bbox_dir", type=str, help="gt images with bbox of synthetic airplanes",
    #                     default='/data/users/yang/data/synthetic_data/Airplanes_gt_bbox/{}/')

    parser.add_argument("--syn_images_save_dir", type=str, help="rgb images of synthetic airplanes",
                        default='/data/users/yang/data/xView_YOLO/images/{}_{}/')
    parser.add_argument("--syn_annos_save_dir", type=str, help="gt of synthetic airplanes",
                        default='/data/users/yang/data/xView_YOLO/labels/{}/{}_{}_cls_xcycwh/')
    parser.add_argument("--syn_txt_save_dir", type=str, help="gt related files of synthetic airplanes",
                        default='/data/users/yang/data/xView_YOLO/labels/{}/{}_{}_cls/')

    parser.add_argument("--data_xview_dir", type=str, help="to save data files",
                        default='/data/users/yang/code/yxu-yolov3-xview/data_xview/{}_cls/')

    parser.add_argument("--syn_data_list_dir", type=str, help="to syn data list files",
                        default='/data/users/yang/code/yxu-yolov3-xview/data_xview/{}_{}_cls/')

    parser.add_argument("--results_dir", type=str, help="to save category files",
                        default='/data/users/yang/code/yxu-yolov3-xview/result_output/{}_cls/{}_seed{}/{}/')

    parser.add_argument("--cat_sample_dir", type=str, help="to save figures",
                        default='/data/users/yang/data/xView_YOLO/cat_samples/{}/{}_cls/')

    parser.add_argument("--val_percent", type=float, default=0.20,
                        help="Percent to split into validation (ie .25 = val set is 25% total)")
    parser.add_argument("--font3", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 10}")
    parser.add_argument("-ft2", "--font2", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 13}")

    parser.add_argument("--class_num", type=int, default=1, help="Number of Total Categories")  # 60  6
    parser.add_argument("--seed", type=int, default=17, help="random seed")
    parser.add_argument("--tile_size", type=int, default=608, help="image size")  # 300 416

    parser.add_argument("--min_region", type=int, default=100, help="the smallest #pixels (area) to form an object")
    parser.add_argument("--link_r", type=int, default=15,
                        help="the #pixels between two connected components to be grouped")

    parser.add_argument("--resolution", type=float, default=0.3, help="resolution of synthetic data")

    syn_args = parser.parse_args()
    syn_args.data_xview_dir = syn_args.data_xview_dir.format(syn_args.class_num)
    syn_args.cat_sample_dir = syn_args.cat_sample_dir.format(syn_args.tile_size, syn_args.class_num)
    return syn_args


def check_prd_gt_iou_xview_syn(data_file, model_id, rare_class, cmt, prefix, res_folder, base_pxwhrs='px23whr3_seed17', hyp_cmt = 'hgiou1_1gpu', seed=17, iou_thres=0.5):
    xview_dir = os.path.join(syn_args.data_xview_dir, base_pxwhrs)
    print('xview_dir', xview_dir)
    data = parse_data_cfg(os.path.join(xview_dir, data_file))
    # fixme--yang.xu
    img_path = data['test']  # path to test images
    img_path = img_path.split('./')[-1]
    lbl_path = data['test_label']
    lbl_path = lbl_path.split('./')[-1]

    df_imgs = pd.read_csv(img_path, header=None)
    df_lbls = pd.read_csv(lbl_path, header=None)
    cinx = cmt.find('model') # first letter index
    endstr = cmt[cinx:]
    rcinx = endstr.rfind('_')
    fstr = endstr[rcinx:] # '_' is included
    sstr = endstr[:rcinx]
    suffix = fstr + '_' + sstr
    print('suffix', suffix)
    # print('res+_folder', res_folder)
    # exit(0)
    if model_id:
        result_path = syn_args.results_dir.format(syn_args.class_num, cmt, seed, res_folder)
    #    print('result_path ', result_path)
    # if len(lcmt) > 1:
    #     suffix = lcmt[1] + '_' + lcmt[0]
    #     result_path = syn_args.results_dir.format(syn_args.class_num, cmt, seed, res_folder.format(hyp_cmt, seed))
    # else:
    #     suffix = 'model{}'.format(model_id)
    #     result_path = syn_args.results_dir.format(syn_args.class_num, cmt, seed, res_folder.format(hyp_cmt, seed, model_id))
#    json_name = prefix + suffix + '*.json'
    json_name = 'results*.json'
#    print('result_path ', result_path)
#    print('json_name', json_name)
    print(os.path.join(result_path, json_name))
    res_json_files = glob.glob(os.path.join(result_path, json_name))
    print('res_json_files ', len(res_json_files))
    if not len(res_json_files):
        return
    res_json_files.sort()
    res_json_file = res_json_files[0]
    res_json = json.load(open(res_json_file))
    
    result_iou_check_dir = os.path.join(syn_args.cat_sample_dir, 'result_iou_check', 'RC{}'.format(rare_class),  cmt, res_folder)
    if not os.path.exists(result_iou_check_dir):
        os.makedirs(result_iou_check_dir)
    img_names = []
    for ix, f in enumerate(df_imgs.loc[:, 0]):
        image_name = os.path.basename(f)
        img_names.append(image_name)
        lbl_file = df_lbls.loc[ix, 0]

        img = cv2.imread(os.path.join(f))
        img_size = img.shape[0]
        good_gt_list = []
        if pps.is_non_zero_file(lbl_file):
            gt_cat = pd.read_csv(lbl_file, header=None, sep=' ')
            gt_cat = gt_cat.to_numpy()
            gt_cat[:, 1:5] = gt_cat[:, 1:5] * img_size
            gt_cat[:, 1] = gt_cat[:, 1] - gt_cat[:, 3] / 2
            gt_cat[:, 2] = gt_cat[:, 2] - gt_cat[:, 4] / 2
            gt_cat[:, 3] = gt_cat[:, 1] + gt_cat[:, 3]
            gt_cat[:, 4] = gt_cat[:, 2] + gt_cat[:, 4]
#            gt_cat = gt_cat[gt_cat[:, -1] == model_id]
            gt_cat = gt_cat[gt_cat[:, -1] == rare_class]
            good_gt_list = gt_cat.tolist()

        result_list = []

        for ri in res_json:
            if ri['image_name'] == image_name:  # ri['image_id'] in rare_img_id_list and
                result_list.append(ri)
        if len(good_gt_list):        
            for gx in range(gt_cat.shape[0]):
            # for gx in good_gt_list:
                g = gt_cat[gx]
                g_bbx = [int(x) for x in g[1:]]
                img = cv2.rectangle(img, (g_bbx[0], g_bbx[1]), (g_bbx[2], g_bbx[3]), (0, 255, 255), 2)  # yellow
                if len(g_bbx) == 5:
                    cv2.putText(img, text='{}'.format(g_bbx[4]), org=(g_bbx[0] + 10, g_bbx[1] + 20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 255))  # yellow
            p_iou = {}  # to keep the larger iou

        for px, p in enumerate(result_list):
            # fixme
            # if p['image_id'] == img_id and p['score'] >= score_thres:
            p['bbox'][2] = p['bbox'][0] + p['bbox'][2]
            p['bbox'][3] = p['bbox'][1] + p['bbox'][3]
            p_bbx = [int(x) for x in p['bbox']]
            p_cat_id = p['category_id']
            p_score = p['score']
            img = cv2.rectangle(img, (p_bbx[0], p_bbx[1]), (p_bbx[2], p_bbx[3]), (255, 255, 0), 2)
 #           cv2.putText(img, text='[conf:{:.3f}]'.format(p_score), org=(p_bbx[2] + 10, p_bbx[3]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(255, 255, 0))  # cyan
            cv2.putText(img, text='[conf:{:.3f}]'.format(p_score), org=(p_bbx[0] , p_bbx[1]-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(255, 255, 0))  # cyan

#            for g in gt_cat:
#                g_bbx = [int(x) for x in g[1:]]
#                iou = coord_iou(p_bbx, g_bbx)
#
#                if iou >= iou_thres:
#                    print('iou---------------------------------->', iou)
#                    print('gbbx', g_bbx)
#                    if px not in p_iou.keys():  # **********keep the largest iou
#                        p_iou[px] = iou
#                    elif iou > p_iou[px]:
#                        p_iou[px] = iou
#                    img = cv2.rectangle(img, (p_bbx[0], p_bbx[1]), (p_bbx[2], p_bbx[3]), (255, 255, 0), 2)
#                    cv2.putText(img, text='[conf:{:.3f}, iou:{:.3f}]'.format(p_score, p_iou[px]), org=(p_bbx[2] + 10, p_bbx[3]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(255, 255, 0))  # cyan
#        if gt_cat.shape[0]:
            #print('image_name', image_name)
        cv2.imwrite(os.path.join(result_iou_check_dir, image_name), img)

if __name__ == "__main__":

    syn_args = get_part_syn_args()

    seed = 17
    iou_thres=0.5
 
#    comments = ['syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias2.5_RC1_v55_color']
    #model_id = 4
    #rare_class = 1
#    comments = ['syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias-0.5_RC5_v22_color']
#    model_id = 5
#    rare_class = 5 
#    comments = ['syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_size_bias0_RC3_v24_color']
#    model_id = 5
#    rare_class = 3
#    comments = ['syn_xview_bkg_px23whr3_xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_dynmu_size_bias0_RC2_v25_color']
#    model_id = 1
#    rare_class = 2
#    comments = ['syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_size_bias0_RC4_v26_color']
#    model_id = 5
#    rare_class = 4
#    hyp_cmt = 'hgiou1_1gpu_5every_val_syn'
#    res_folder = 'test_on_ori_nrcbkg_aug_rc_{}_m{}_rc{}_{}_iou50_epochs_v1'

    ############### dynsigma size
#    comments = ['syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_bxmuller_size_bias0_RC5_v1']
#    model_id = 5
#    rare_class = 5
    comments = ['syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_bxmuller_size_bias0.12_RC4_v5']
    model_id = 5
    rare_class = 4   
    apN = 50
    prefix = 'results_syn_iou{}'.format(apN)
    eht = 'easy'

    hyp_cmt = 'hgiou1_1gpu_val_syn'
    res_folder = 'test_on_ori_nrcbkg_aug_rc_{}_m{}_rc{}_{}_iou{}_epochs'
    data_file = 'xview_ori_nrcbkg_aug_rc_test_{}_m{}_rc{}_{}.data'
##
    base_pxwhrs = 'px23whr3_seed17'

#    data_file = 'xviewtest_{}_m{}_rc{}_{}.data'
#    res_folder = 'test_on_xview_{}_m{}_rc{}_{}'
    #res_folder = 'test_on_xview_{}_m{}_rc{}_{}_iou20'
    

#    data_file = 'xviewtest_{}_m{}_rc{}_{}_aug.data'
#    res_folder = 'test_on_xview_{}_m{}_rc{}_{}_aug'
    
#    data_file = 'xviewtest_{}_upscale_m{}_rc{}_{}.data'
#    res_folder = 'test_on_xview_{}_upscale_m{}_rc{}_{}'

#    data_file = 'xviewtest_{}_m{}_rc{}_2315.data'.format(base_pxwhrs, model_id, rare_class)
#    res_folder = 'test_on_xview_with_model_{}_2315_hard'.format(hyp_cmt)
    for eh in ['hard', 'easy']:
        d_file = data_file.format(base_pxwhrs, model_id, rare_class, eh)
        r_folder = res_folder.format(hyp_cmt, model_id, rare_class, eh, apN)
        for cmt in comments:
            check_prd_gt_iou_xview_syn(d_file, model_id, rare_class, cmt, prefix, r_folder, base_pxwhrs=base_pxwhrs, hyp_cmt=hyp_cmt, seed=seed, iou_thres=iou_thres)
