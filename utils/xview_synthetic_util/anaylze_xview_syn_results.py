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
import seaborn as sn

from utils.data_process_distribution_vis_util import process_wv_coco_for_yolo_patches_no_trnval as pwv
from utils.utils_xview import coord_iou
from utils.xview_synthetic_util import preprocess_synthetic_data_distribution as pps

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


def get_val_imgid_by_name(name):
    val_files = pd.read_csv(os.path.join(syn_args.data_xview_dir, 'xviewval_img.txt'), header=None).to_numpy()
    # print('val_files 0', val_files[0])
    val_names = [os.path.basename(vf[0]) for vf in val_files]
    img_id = val_names.index(name)
    return img_id


def check_prd_gt_iou_xview_syn(dt, sr, image_name, score_thres=0.3, iou_thres=0.5, px_thres=6, whr_thres=4):
    '''
    Note that there is possible some lower iou may cover the lager iou computed previously, remember to keep the larger iou
    :param image_name:
    :param score_thres:
    :param iou_thres:
    :param px_thres:
    :param whr_thres:
    :return:
    '''
    args = pwv.get_args()
    results_dir = syn_args.results_dir.format(syn_args.class_num, dt, sr)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # fixme
    # img_id = get_val_imgid_by_name(image_name)
    img = cv2.imread(args.images_save_dir + image_name)
    img_size = img.shape[0]
    gt_cat = pd.read_csv(args.annos_save_dir + image_name.replace('.jpg', TXT_SUFFIX), header=None, delimiter=' ')
    gt_cat = gt_cat.to_numpy()
    gt_cat[:, 1:] = gt_cat[:, 1:] * img_size
    gt_cat[:, 1] = gt_cat[:, 1] - gt_cat[:, 3] / 2
    gt_cat[:, 2] = gt_cat[:, 2] - gt_cat[:, 4] / 2
    gt_cat[:, 3] = gt_cat[:, 1] + gt_cat[:, 3]
    gt_cat[:, 4] = gt_cat[:, 2] + gt_cat[:, 4]
    good_gt_list = []
    for gx in range(gt_cat.shape[0]):
        w, h = gt_cat[gx, 3] - gt_cat[gx, 1], gt_cat[gx, 4] - gt_cat[gx, 2]
        whr = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        if whr <= whr_thres and w >= px_thres and h >= px_thres:
            good_gt_list.append(gt_cat[gx, :])

    result_json_file = results_dir + 'results_{}_{}.json'.format(dt, sr)
    result_allcat_list = json.load(open(result_json_file))
    result_list = []
    # #fixme filter, and rare_result_allcat_list contains rare_cat_ids, rare_img_id_list and object score larger than score_thres
    for ri in result_allcat_list:
        if ri['image_name'] == image_name and ri['score'] >= score_thres:  # ri['image_id'] in rare_img_id_list and
            result_list.append(ri)
    # print('len result_list', len(result_list))
    del result_allcat_list

    for g in good_gt_list:
        g_bbx = [int(x) for x in g[1:]]
        img = cv2.rectangle(img, (g_bbx[0], g_bbx[1]), (g_bbx[2], g_bbx[3]), (0, 255, 255), 2)  # yellow

    p_iou = {}  # to keep the larger iou

    for px, p in enumerate(result_list):
        # fixme
        # if p['image_id'] == img_id and p['score'] >= score_thres:
        w, h = p['bbox'][2:]
        whr = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        if whr > whr_thres or w < px_thres or h < px_thres:  #
            continue
            # print('p-bbx ', p_bbx)
        p['bbox'][2] = p['bbox'][0] + p['bbox'][2]
        p['bbox'][3] = p['bbox'][1] + p['bbox'][3]
        p_bbx = [int(x) for x in p['bbox']]
        p_cat_id = p['category_id']
        for g in good_gt_list:
            g_bbx = [int(x) for x in g[1:]]
            iou = coord_iou(p_bbx, g_bbx)

            if iou >= iou_thres:
                print('iou---------------------------------->', iou)
                print('gbbx', g_bbx)
                if px not in p_iou.keys():  # **********keep the largest iou
                    p_iou[px] = iou
                elif iou > p_iou[px]:
                    p_iou[px] = iou

                img = cv2.rectangle(img, (p_bbx[0], p_bbx[1]), (p_bbx[2], p_bbx[3]), (255, 255, 0), 2)
                cv2.putText(img, text='[{}, {:.3f}]'.format(p_cat_id, p_iou[px]), org=(p_bbx[0] - 10, p_bbx[1] - 10),
                            # [pr_bx[0], pr[-1]]
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(255, 255, 0))  # cyan

    result_iou_check_dir = args.cat_sample_dir + 'result_iou_check/{}_{}/'.format(dt, sr)
    if not os.path.exists(result_iou_check_dir):
        os.mkdir(result_iou_check_dir)
    cv2.imwrite(result_iou_check_dir + image_name, img)

    return [v for v in p_iou.values()]


def plot_val_results_iou_comp():
    val_iou_path = os.path.join(args.txt_save_dir, 'val_result_iou_map')

    display_type = ['syn']
    syn_ratio = [0]
    syn0_iou_json_file = os.path.join(val_iou_path, 'xViewval_syn_0_iou.json')
    syn0_iou_map = json.load(open(syn0_iou_json_file))
    syn0_iou_list = []
    for v in syn0_iou_map.values():
        if len(v):
            syn0_iou_list.extend(v)

    colors = ['orange', 'm', 'g']
    display_type = ['syn_texture', 'syn_color', 'syn_mixed'] #, 'syn_texture0', 'syn_color0']
    syn_ratio = [0.25, 0.5, 0.75]
    fig, axs = plt.subplots(3, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ix, sr in enumerate(syn_ratio):
        for jx, dt in enumerate(display_type):
            iou_json_file = os.path.join(val_iou_path, 'xViewval_{}_{}_iou.json'.format(dt, sr))
            iou_map = json.load(open(iou_json_file))
            iou_list = []
            for v in iou_map.values():
                if len(v):
                    iou_list.extend(v)
            axs[ix, jx].hist(syn0_iou_list, bins=10, histtype="bar", alpha=0.75, density=True, label='syn_0')
            axs[ix, jx].hist(iou_list, bins=10, histtype="bar", alpha=0.75, color=colors[jx], density=True, label='{}_{}'.format(dt, sr)) # facecolor='g',
            axs[ix, jx].grid(True)
            axs[ix, jx].legend()
            axs[0, jx].set_title(dt)
            axs[2, jx].set_xlabel('IOU')
        axs[ix, 0].set_ylabel('Number of Images')
    fig.suptitle('Val Results IoU Comparison', fontsize=20)
    save_dir = os.path.join(val_iou_path, 'figures')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    fig.savefig(os.path.join(save_dir, 'val_result_iou_comp.png'))
    fig.show()


def get_fp_fn_list_airplane(dt, sr, catid=0, iou_thres=0.5, score_thres=0.3, px_thres=6, whr_thres=4):
    ''' ground truth '''
    results_dir = syn_args.results_dir.format(syn_args.class_num, dt, sr)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    result_json_file = results_dir + 'results_{}_{}.json'.format(dt, sr)
    result_allcat_list = json.load(open(result_json_file))
    result_list = []
    # #fixme filter, and rare_result_allcat_list contains rare_cat_ids, rare_img_id_list and object score larger than score_thres
    for ri in result_allcat_list:
        if ri['category_id'] == catid and ri['score'] >= score_thres:  # ri['image_id'] in rare_img_id_list and
            result_list.append(ri)
    print('len result_list', len(result_list))
    del result_allcat_list

    val_labels = pd.read_csv(os.path.join(syn_args.data_xview_dir, 'xviewval_lbl.txt'), header=None)
    img_name_2_fp_list_maps = {}
    img_name_2_fn_list_maps = {}
    for ix, vl in enumerate(val_labels.iloc[:, 0]):
        img_name = os.path.basename(vl).replace(TXT_SUFFIX, IMG_SUFFIX)
        # if img_name == '1585_2.jpg':
        #     print(img_name)
        # if img_name =='2518_2.jpg':
        #     print(img_name)
        img_name_2_fp_list_maps[img_name] = []
        img_name_2_fn_list_maps[img_name] = []
        good_gt_list = []
        df_lbl = pd.read_csv(vl, header=None, delimiter=' ')
        df_lbl.iloc[:, 1:] = df_lbl.iloc[:, 1:] * syn_args.tile_size
        df_lbl.iloc[:, 1] = df_lbl.iloc[:, 1] - df_lbl.iloc[:, 3] / 2
        df_lbl.iloc[:, 2] = df_lbl.iloc[:, 2] - df_lbl.iloc[:, 4] / 2
        df_lbl.iloc[:, 3] = df_lbl.iloc[:, 1] + df_lbl.iloc[:, 3]
        df_lbl.iloc[:, 4] = df_lbl.iloc[:, 2] + df_lbl.iloc[:, 4]
        df_lbl = df_lbl.to_numpy()
        for dx in range(df_lbl.shape[0]):
            w, h = df_lbl[dx, 3] - df_lbl[dx, 1], df_lbl[dx, 4] - df_lbl[dx, 2]
            whr = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            if whr <= whr_thres and w >= px_thres and h >= px_thres:
                good_gt_list.append(df_lbl[dx, :])
        gt_boxes = []
        if good_gt_list:
            good_gt_arr = np.array(good_gt_list)
            gt_boxes = good_gt_arr[:, 1:]  # x1y1x2y2
            gt_classes = good_gt_arr[:, 0]

        prd_list = [rx for rx in result_list if rx['image_name'] == img_name]
        prd_lbl_list = []
        for rx in prd_list:
            w, h = rx['bbox'][2:]
            whr = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            rx['bbox'][2] = rx['bbox'][2] + rx['bbox'][0]
            rx['bbox'][3] = rx['bbox'][3] + rx['bbox'][1]
            # print('whr', whr)
            if whr <= whr_thres and w >= px_thres and h >= px_thres and rx['score'] >= score_thres:  #
                prd_lbl = [rx['category_id']]
                prd_lbl.extend([int(b) for b in rx['bbox']])  # x1y1x2y2
                prd_lbl.extend([rx['score']])
                prd_lbl_list.append(prd_lbl)

        matches = []
        dt_boxes = []
        if prd_lbl_list:
            prd_lbl_arr = np.array(prd_lbl_list)
            # print(prd_lbl_arr.shape)
            dt_scores = prd_lbl_arr[:, -1]  # [prd_lbl_arr[:, -1] >= score_thres]
            dt_boxes = prd_lbl_arr[dt_scores >= score_thres][:, 1:-1]
            dt_classes = prd_lbl_arr[dt_scores >= score_thres][:, 0]

        for i in range(len(gt_boxes)):
            for j in range(len(dt_boxes)):
                iou = coord_iou(gt_boxes[i], dt_boxes[j])

                if iou >= iou_thres:
                    matches.append([i, j, iou])

        matches = np.array(matches)

        if matches.shape[0] > 0:
            # Sort list of matches by descending IOU so we can remove duplicate detections
            # while keeping the highest IOU entry.
            matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

            # Remove duplicate detections from the list.
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

            # Sort the list again by descending IOU. Removing duplicates doesn't preserve
            # our previous sort.
            matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

            # Remove duplicate ground truths from the list.
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

        for i in range(len(gt_boxes)):
            if matches.shape[0] > 0 and matches[matches[:, 0] == i].shape[0] == 1:
                mt_i = matches[matches[:, 0] == i]
                # print('mt_i', mt_i.shape)
            else:
                # fixme
                # unique matches at most has one match for each ground truth
                # 1. ground truth id deleted due to duplicate detections  --> FN
                # 2. matches.shape[0] == 0 --> no matches --> FN
                c_box_iou = [gt_classes[i]]
                c_box_iou.extend(gt_boxes[i])
                c_box_iou.append(0)  # [cat_id, box[0:4], iou]
                img_name_2_fn_list_maps[img_name].append(c_box_iou)

        for j in range(len(dt_boxes)):
            # fixme
            # detected object not in the matches --> FP
            # 1. deleted due to duplicate ground truth (background-->Y_prd)
            # 2. lower than iou_thresh (maybe iou=0)  (background-->Y_prd)
            # because in the inference phrase, we do not know the ground truth, we cannot get the IoU
            # we determine the prediction by the object score
            if matches.shape[0] > 0 and matches[matches[:, 1] == j].shape[0] == 0:
                c_box_iou = [dt_classes[j]]
                c_box_iou.extend(dt_boxes[j])
                c_box_iou.append(0)  # [cat_id, box[0:4], iou]
                # print(c_box_iou)
                img_name_2_fp_list_maps[img_name].append(c_box_iou)
            elif matches.shape[0] == 0:  # fixme
                c_box_iou = [dt_classes[j]]
                c_box_iou.extend(dt_boxes[j])
                c_box_iou.append(0)  # [cat_id, box[0:4], iou]
                img_name_2_fp_list_maps[img_name].append(c_box_iou)

    save_dir = os.path.join(args.txt_save_dir,
                            'val_img_2_fp_fn_list/{}_{}/'.format(dt, sr))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fp_json_file = os.path.join(save_dir,
                                'xViewval_{}_{}_img_2_fp_maps.json'.format(dt, sr))  # topleft
    json.dump(img_name_2_fp_list_maps, open(fp_json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)

    fn_json_file = os.path.join(save_dir,
                                'xViewval_{}_{}_img_2_fn_maps.json'.format(dt, sr))  # topleft
    json.dump(img_name_2_fn_list_maps, open(fn_json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)


def plot_val_img_with_fp_fn_bbox(dt, sr):
    fp_fn_list_dir = os.path.join(args.txt_save_dir,
                                  'val_img_2_fp_fn_list/{}_{}/'.format(dt, sr))
    img_fp_fn_bbox_path = os.path.join(args.cat_sample_dir,
                                       'val_img_with_fp_fn_bbox/{}_{}/'.format(dt,
                                                                               sr))
    if not os.path.exists(img_fp_fn_bbox_path):
        os.makedirs(img_fp_fn_bbox_path)

    fp_maps = json.load(open(os.path.join(fp_fn_list_dir,
                                          'xViewval_{}_{}_img_2_fp_maps.json'.format(dt,
                                                                                     sr))))
    fn_maps = json.load(open(os.path.join(fp_fn_list_dir,
                                          'xViewval_{}_{}_img_2_fn_maps.json'.format(dt,
                                                                                     sr))))
    fp_color = (0, 0, 255)  # Red
    fn_color = (255, 0, 0)  # Blue
    img_names = [k for k in fp_maps.keys()]
    for name in img_names:
        img = cv2.imread(os.path.join(args.images_save_dir, name))
        fp_list = fp_maps[name]
        fn_list = fn_maps[name]
        if not fp_list and not fn_list:
            continue
        for gx, gr in enumerate(fn_list):
            gr_bx = [int(x) for x in gr[1:-1]]
            img = cv2.rectangle(img, (gr_bx[0], gr_bx[1]), (gr_bx[2], gr_bx[3]), fn_color, 2)  # w1, h1, w2, h2
            # img = pwv.drawrect(img, (gr_bx[0], gr_bx[1]), (gr_bx[2], gr_bx[3]), fn_color, thickness=2, style='dotted')
            cv2.putText(img, text=str(int(gr[0])), org=(gr_bx[2] - 8, gr_bx[3] - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 255))
        for px, pr in enumerate(fp_list):
            pr_bx = [int(x) for x in pr[1:-1]]
            img = cv2.rectangle(img, (pr_bx[0], pr_bx[1]), (pr_bx[2], pr_bx[3]), fp_color, 2)
            # img = pwv.drawrect(img, (pr_bx[0], pr_bx[1]), (pr_bx[2], pr_bx[3]), fp_color, thickness=1, style='dotted')
            cv2.putText(img, text='{}'.format(int(pr[0])), org=(pr_bx[0] + 5, pr_bx[1] + 8),  # [pr_bx[0], pr[-1]]
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 0))
        cv2.imwrite(os.path.join(img_fp_fn_bbox_path, name), img)


def plot_val_img_with_gt_prd_bbox(dt, sr, catid=0, score_thres=0.3, px_thres=6, whr_thres=4):
    gt_prd_bbx_save_dir = os.path.join(args.cat_sample_dir,
                                       'val_img_with_gt_prd_objscore_bbx/{}_{}/'.format(dt, sr))
    if not os.path.exists(gt_prd_bbx_save_dir):
        os.makedirs(gt_prd_bbx_save_dir)

    results_dir = syn_args.results_dir.format(syn_args.class_num, dt, sr)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    result_json_file = results_dir + 'results_{}_{}.json'.format(dt, sr)
    result_allcat_list = json.load(open(result_json_file))
    result_list = []
    # #fixme filter, and rare_result_allcat_list contains rare_cat_ids, rare_img_id_list and object score larger than score_thres
    for ri in result_allcat_list:
        if ri['category_id'] == catid and ri['score'] >= score_thres:  # ri['image_id'] in rare_img_id_list and
            result_list.append(ri)
    print('len result_list', len(result_list))
    del result_allcat_list

    prd_color = (255, 255, 0)
    gt_color = (0, 255, 255)  # yellow

    val_labels = pd.read_csv(os.path.join(args.data_save_dir, 'xviewval_lbl.txt'), header=None)
    for ix, vl in enumerate(val_labels.iloc[:, 0]):
        img_name = os.path.basename(vl).replace(TXT_SUFFIX, IMG_SUFFIX)
        img = cv2.imread((os.path.join(args.images_save_dir, img_name)))
        good_gt_list = []
        df_lbl = pd.read_csv(vl, header=None, delimiter=' ')
        df_lbl.iloc[:, 1:] = df_lbl.iloc[:, 1:] * syn_args.tile_size
        df_lbl.iloc[:, 1] = df_lbl.iloc[:, 1] - df_lbl.iloc[:, 3] / 2
        df_lbl.iloc[:, 2] = df_lbl.iloc[:, 2] - df_lbl.iloc[:, 4] / 2
        df_lbl.iloc[:, 3] = df_lbl.iloc[:, 1] + df_lbl.iloc[:, 3]
        df_lbl.iloc[:, 4] = df_lbl.iloc[:, 2] + df_lbl.iloc[:, 4]
        df_lbl = df_lbl.to_numpy()
        for dx in range(df_lbl.shape[0]):
            w, h = df_lbl[dx, 3] - df_lbl[dx, 1], df_lbl[dx, 4] - df_lbl[dx, 2]
            whr = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            if whr <= whr_thres and w >= px_thres and h >= px_thres:
                good_gt_list.append(df_lbl[dx, :])

        # fixme
        # prd_list = [rx for rx in result_list if rx['image_id'] == ix]
        prd_list = [rx for rx in result_list if rx['image_name'] == img_name]
        prd_lbl_list = []
        for rx in prd_list:
            w, h = rx['bbox'][2:]
            whr = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            rx['bbox'][2] = rx['bbox'][2] + rx['bbox'][0]
            rx['bbox'][3] = rx['bbox'][3] + rx['bbox'][1]
            # print('whr', whr)
            if whr <= whr_thres and w >= px_thres and h >= px_thres and rx['score'] >= score_thres:  #
                prd_lbl = [rx['category_id']]
                prd_lbl.extend([int(b) for b in rx['bbox']])  # x1y1x2y2
                prd_lbl.extend([rx['score']])
                prd_lbl_list.append(prd_lbl)

        for gt in good_gt_list:
            gt_bbox = [int(x) for x in gt[1:]]
            img = cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), gt_color, 2)  # w1, h1, w2, h2
            # img = pwv.drawrect(img, (gr_bx[0], gr_bx[1]), (gr_bx[2], gr_bx[3]), gt_color, thickness=2, style='dotted')
            cv2.putText(img, text=str(int(gt[0])), org=(gt_bbox[2] - 8, gt_bbox[3] - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=gt_color)

        for px, pr in enumerate(prd_lbl_list):
            pr_bx = [int(x) for x in pr[1:-1]]
            img = cv2.rectangle(img, (pr_bx[0], pr_bx[1]), (pr_bx[2], pr_bx[3]), prd_color, 2)
            # img = pwv.drawrect(img, (pr_bx[0], pr_bx[1]), (pr_bx[2], pr_bx[3]), prd_color, thickness=1, style='dotted')
            cv2.putText(img, text='{:.2f}'.format(pr[-1]), org=(pr_bx[0] - 5, pr_bx[1] - 8),  # [pr_bx[0], pr[-1]]
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=prd_color)
        cv2.imwrite(os.path.join(gt_prd_bbx_save_dir, img_name), img)


def draw_bar_compare_fp_fn_number_of_different_syn_ratio():
    fp_fn_0_dir = os.path.join(args.txt_save_dir, 'val_img_2_fp_fn_list/syn_0/')
    fp_0_file = json.load(open(os.path.join(fp_fn_0_dir, 'xViewval_syn_0_img_2_fp_maps.json')))
    fn_0_file = json.load(open(os.path.join(fp_fn_0_dir, 'xViewval_syn_0_img_2_fn_maps.json')))
    fp_0_num = len([v for v in fp_0_file.values() if v])
    fn_0_num = len([v for v in fn_0_file.values() if v])

    save_dir = os.path.join(args.txt_save_dir, 'val_img_2_fp_fn_list/figures/')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    x = [1, 3, 5, 7, 9, 11]
    xlabels = ['FP ratio=0.25', 'FP ratio=0.5', 'FP ratio=0.75', 'FN ratio=0.25', 'FN ratio=0.5', 'FN ratio=0.75']
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    fig, ax = plt.subplots(1, 1)
    width = 0.3
    syn_ratios = [0.25, 0.5, 0.75]
    for ix, r in enumerate(syn_ratios):
        fp_fn_tx_path = os.path.join(args.txt_save_dir, 'val_img_2_fp_fn_list/{}_{}/'.format('syn_texture', r))
        fp_tx_file = json.load(
            open(os.path.join(fp_fn_tx_path, 'xViewval_{}_{}_img_2_fp_maps.json'.format('syn_texture', r))))
        fn_tx_file = json.load(
            open(os.path.join(fp_fn_tx_path, 'xViewval_{}_{}_img_2_fn_maps.json'.format('syn_texture', r))))
        fp_tx_num = len([k for k in fp_tx_file.keys() if fp_tx_file.get(k)])
        fn_tx_num = len([k for k in fn_tx_file.keys() if fn_tx_file.get(k)])

        fp_fn_clr_path = os.path.join(args.txt_save_dir, 'val_img_2_fp_fn_list/{}_{}/'.format('syn_color', r))
        fp_clr_file = json.load(
            open(os.path.join(fp_fn_clr_path, 'xViewval_{}_{}_img_2_fp_maps.json'.format('syn_color', r))))
        fn_clr_file = json.load(
            open(os.path.join(fp_fn_clr_path, 'xViewval_{}_{}_img_2_fn_maps.json'.format('syn_color', r))))
        fp_clr_num = len([k for k in fp_clr_file.keys() if fp_clr_file.get(k)])
        fn_clr_num = len([k for k in fn_clr_file.keys() if fn_clr_file.get(k)])

        fp_fn_mx_path = os.path.join(args.txt_save_dir, 'val_img_2_fp_fn_list/{}_{}/'.format('syn_mixed', r))
        fp_mx_file = json.load(
            open(os.path.join(fp_fn_mx_path, 'xViewval_{}_{}_img_2_fp_maps.json'.format('syn_mixed', r))))
        fn_mx_file = json.load(
            open(os.path.join(fp_fn_mx_path, 'xViewval_{}_{}_img_2_fn_maps.json'.format('syn_mixed', r))))
        fp_mx_num = len([k for k in fp_mx_file.keys() if fp_mx_file.get(k)])
        fn_mx_num = len([k for k in fn_mx_file.keys() if fn_mx_file.get(k)])

        rects_syn_0 = ax.bar([x[ix] - width, x[ix + 3] - width], [fp_0_num, fn_0_num], width, label='syn_ratio=0')
        autolabel(ax, rects_syn_0, x, xlabels, [fp_0_num, fn_0_num], rotation=0)

        rects_syn_clr = ax.bar([x[ix], x[ix + 3]], [fp_clr_num, fn_clr_num], width,
                               label='syn_color_ratio={}'.format(r))  # , label=labels
        autolabel(ax, rects_syn_clr, x, xlabels, [fp_clr_num, fn_clr_num], rotation=0)

        rects_syn_tx = ax.bar([x[ix] + width, x[ix + 3] + width], [fp_tx_num, fn_tx_num], width,
                              label='syn_texture_ratio={}'.format(r))  # , label=labels
        autolabel(ax, rects_syn_tx, x, xlabels, [fp_tx_num, fn_tx_num], rotation=0)

        rects_syn_mx = ax.bar([x[ix] + 2 * width, x[ix + 3] + 2 * width], [fp_mx_num, fn_mx_num], width,
                              label='syn_mixed_ratio={}'.format(r))  # , label=labels
        autolabel(ax, rects_syn_mx, x, xlabels, [fp_mx_num, fn_mx_num], rotation=0)

    ax.legend()
    ylabel = "Number"
    plt.title('', literal_eval(syn_args.font2))
    plt.ylabel(ylabel, literal_eval(syn_args.font2))
    plt.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'cmp_fp_fn_syn0_vs_syn_clr_tx_mx.jpg'.format(r)))
    plt.show()


def draw_bar_compare_fp_fn_number_by_syn_ratio(r):
    fp_fn_0_dir = os.path.join(args.txt_save_dir, 'val_img_2_fp_fn_list/syn_0/')
    fp_0_file = json.load(open(os.path.join(fp_fn_0_dir, 'xViewval_syn_0_img_2_fp_maps.json')))
    fn_0_file = json.load(open(os.path.join(fp_fn_0_dir, 'xViewval_syn_0_img_2_fn_maps.json')))
    fp_0_num = len([v for v in fp_0_file.values() if v])
    fn_0_num = len([v for v in fn_0_file.values() if v])

    save_dir = os.path.join(args.txt_save_dir, 'val_img_2_fp_fn_list/figures/')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    fp_fn_tx_path = os.path.join(args.txt_save_dir, 'val_img_2_fp_fn_list/{}_{}/'.format('syn_texture', r))
    fp_tx_file = json.load(
        open(os.path.join(fp_fn_tx_path, 'xViewval_{}_{}_img_2_fp_maps.json'.format('syn_texture', r))))
    fn_tx_file = json.load(
        open(os.path.join(fp_fn_tx_path, 'xViewval_{}_{}_img_2_fn_maps.json'.format('syn_texture', r))))
    fp_tx_num = len([k for k in fp_tx_file.keys() if fp_tx_file.get(k)])
    fn_tx_num = len([k for k in fn_tx_file.keys() if fn_tx_file.get(k)])

    fp_fn_clr_path = os.path.join(args.txt_save_dir, 'val_img_2_fp_fn_list/{}_{}/'.format('syn_color', r))
    fp_clr_file = json.load(
        open(os.path.join(fp_fn_clr_path, 'xViewval_{}_{}_img_2_fp_maps.json'.format('syn_color', r))))
    fn_clr_file = json.load(
        open(os.path.join(fp_fn_clr_path, 'xViewval_{}_{}_img_2_fn_maps.json'.format('syn_color', r))))
    fp_clr_num = len([k for k in fp_clr_file.keys() if fp_clr_file.get(k)])
    fn_clr_num = len([k for k in fn_clr_file.keys() if fn_clr_file.get(k)])

    fp_fn_mx_path = os.path.join(args.txt_save_dir, 'val_img_2_fp_fn_list/{}_{}/'.format('syn_mixed', r))
    fp_mx_file = json.load(
        open(os.path.join(fp_fn_mx_path, 'xViewval_{}_{}_img_2_fp_maps.json'.format('syn_mixed', r))))
    fn_mx_file = json.load(
        open(os.path.join(fp_fn_mx_path, 'xViewval_{}_{}_img_2_fn_maps.json'.format('syn_mixed', r))))
    fp_mx_num = len([k for k in fp_mx_file.keys() if fp_mx_file.get(k)])
    fn_mx_num = len([k for k in fn_mx_file.keys() if fn_mx_file.get(k)])

    x = [3, 6]
    xlabels = ['FP', 'FN']
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    fig, ax = plt.subplots(1, 1)
    width = 0.35
    rects_syn_0 = ax.bar([x[0] - width, x[1] - width], [fp_0_num, fn_0_num], width, label='syn_ratio=0')
    autolabel(ax, rects_syn_0, x, xlabels, [fp_0_num, fn_0_num], rotation=0)

    rects_syn_clr = ax.bar([x[0], x[1]], [fp_clr_num, fn_clr_num], width,
                           label='syn_color_ratio={}'.format(r))  # , label=labels
    autolabel(ax, rects_syn_clr, x, xlabels, [fp_clr_num, fn_clr_num], rotation=0)

    rects_syn_tx = ax.bar([x[0] + width, x[1] + width], [fp_tx_num, fn_tx_num], width,
                          label='syn_texture_ratio={}'.format(r))  # , label=labels
    autolabel(ax, rects_syn_tx, x, xlabels, [fp_tx_num, fn_tx_num], rotation=0)

    rects_syn_mx = ax.bar([x[0] + 2 * width, x[1] + 2 * width], [fp_mx_num, fn_mx_num], width,
                          label='syn_mixed_ratio={}'.format(r))  # , label=labels
    autolabel(ax, rects_syn_mx, x, xlabels, [fp_mx_num, fn_mx_num], rotation=0)

    ax.legend()
    ylabel = "Number"
    plt.title('', literal_eval(syn_args.font2))
    plt.ylabel(ylabel, literal_eval(syn_args.font2))
    plt.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'cmp_fp_fn_syn0_vs_syn_clr_tx_{}.jpg'.format(r)))
    plt.show()


def look_for_reduced_fp_fn_in_image(syn_display_type, syn_ratio):
    fp_fn_0_dir = os.path.join(args.txt_save_dir, 'val_img_2_fp_fn_list/syn_0/')
    fp_0_file = json.load(open(os.path.join(fp_fn_0_dir, 'xViewval_syn_0_img_2_fp_maps.json')))
    fn_0_file = json.load(open(os.path.join(fp_fn_0_dir, 'xViewval_syn_0_img_2_fn_maps.json')))
    fp_0_list = [k for k in fp_0_file.keys() if fp_0_file.get(k)]
    fn_0_list = [k for k in fn_0_file.keys() if fn_0_file.get(k)]

    fp_fn_tx_path = os.path.join(args.txt_save_dir, 'val_img_2_fp_fn_list/{}_{}/'.format(syn_display_type, syn_ratio))
    fp_tx_file = json.load(
        open(os.path.join(fp_fn_tx_path, 'xViewval_{}_{}_img_2_fp_maps.json'.format(syn_display_type, syn_ratio))))
    fn_tx_file = json.load(
        open(os.path.join(fp_fn_tx_path, 'xViewval_{}_{}_img_2_fn_maps.json'.format(syn_display_type, syn_ratio))))
    fp_tx_list = [k for k in fp_tx_file.keys() if fp_tx_file.get(k)]
    fn_tx_list = [k for k in fn_tx_file.keys() if fn_tx_file.get(k)]

    fp_only_in_syn_path = os.path.join(args.cat_sample_dir, 'reduced_fp_fn',
                                       'fp_only_in_{}_{}_not_in_xview'.format(syn_display_type, syn_ratio))
    if os.path.exists(fp_only_in_syn_path):
        shutil.rmtree(fp_only_in_syn_path)
    os.makedirs(fp_only_in_syn_path)
    fn_only_in_syn_path = os.path.join(args.cat_sample_dir, 'reduced_fp_fn',
                                       'fn_only_in_{}_{}_not_in_xview'.format(syn_display_type, syn_ratio))
    if os.path.exists(fn_only_in_syn_path):
        shutil.rmtree(fn_only_in_syn_path)
    os.makedirs(fn_only_in_syn_path)

    fp_only_in_syn0_path = os.path.join(args.cat_sample_dir, 'reduced_fp_fn',
                                        'fp_only_in_xview_not_in_{}_{}'.format(syn_display_type, syn_ratio))
    fn_only_in_syn0_path = os.path.join(args.cat_sample_dir, 'reduced_fp_fn',
                                        'fn_only_in_xview_not_in_{}_{}'.format(syn_display_type, syn_ratio))
    if os.path.exists(fp_only_in_syn0_path):
        shutil.rmtree(fp_only_in_syn0_path)
    os.makedirs(fp_only_in_syn0_path)
    if os.path.exists(fn_only_in_syn0_path):
        shutil.rmtree(fn_only_in_syn0_path)
    os.makedirs(fn_only_in_syn0_path)
    syn_src_dir = os.path.join(args.cat_sample_dir,
                               'val_img_with_fp_fn_bbox/{}_{}/'.format(syn_display_type,
                                                                       syn_ratio))
    syn0_src_dir = os.path.join(args.cat_sample_dir,
                                'val_img_with_fp_fn_bbox/syn_0/')

    fp_only_in_syn0_list = [p for p in fp_0_list if p not in fp_tx_list]
    fn_only_in_syn0_list = [n for n in fn_0_list if n not in fn_tx_list]
    print('FP only syn0 list', fp_only_in_syn0_list)
    print('FN only syn0 list', fn_only_in_syn0_list)

    for f in fp_only_in_syn0_list:
        shutil.copy(os.path.join(syn0_src_dir, f),
                    os.path.join(fp_only_in_syn0_path, f))
    for f in fn_only_in_syn0_list:
        shutil.copy(os.path.join(syn0_src_dir, f),
                    os.path.join(fn_only_in_syn0_path, f))

    fp_only_in_syn_list = [p for p in fp_tx_list if p not in fp_0_list]
    fn_only_in_syn_list = [n for n in fn_tx_list if n not in fn_0_list]
    print('FP only in syn list', fp_only_in_syn_list)
    print('FN only in syn list', fn_only_in_syn_list)
    for f in fp_only_in_syn_list:
        shutil.copy(os.path.join(syn_src_dir, f),
                    os.path.join(fp_only_in_syn_path, f))
    for f in fn_only_in_syn_list:
        shutil.copy(os.path.join(syn_src_dir, f),
                    os.path.join(fn_only_in_syn_path, f))


def get_part_syn_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_filepath", type=str, help="Filepath to GEOJSON coordinate file",
                        default='/media/lab/Yang/data/xView/xView_train.geojson')

    parser.add_argument("--syn_plane_img_anno_dir", type=str, help="images and annotations of synthetic airplanes",
                        default='/media/lab/Yang/data/synthetic_data/Airplanes/{}/')

    parser.add_argument("--syn_plane_txt_dir", type=str, help="txt labels of synthetic airplanes",
                        default='/media/lab/Yang/data/synthetic_data/Airplanes_txt_xcycwh/{}/')

    parser.add_argument("--syn_plane_gt_bbox_dir", type=str, help="gt images with bbox of synthetic airplanes",
                        default='/media/lab/Yang/data/synthetic_data/Airplanes_gt_bbox/{}/')

    parser.add_argument("--syn_images_save_dir", type=str, help="rgb images of synthetic airplanes",
                        default='/media/lab/Yang/data/xView_YOLO/images/{}_{}/')
    parser.add_argument("--syn_annos_save_dir", type=str, help="gt of synthetic airplanes",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_{}_cls_xcycwh/')
    parser.add_argument("--syn_txt_save_dir", type=str, help="gt related files of synthetic airplanes",
                        default='/media/lab/Yang/data/xView_YOLO/labels/{}/{}_{}_cls/')

    parser.add_argument("--data_xview_dir", type=str, help="to save data files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_cls/')

    parser.add_argument("--syn_data_list_dir", type=str, help="to syn data list files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_{}_cls/')

    parser.add_argument("--results_dir", type=str, help="to save category files",
                        default='/media/lab/Yang/code/yolov3/result_output/{}_cls/{}_{}/')

    parser.add_argument("--cat_sample_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/')

    parser.add_argument("--val_percent", type=float, default=0.20,
                        help="Percent to split into validation (ie .25 = val set is 25% total)")
    parser.add_argument("--font3", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 10}")
    parser.add_argument("-ft2", "--font2", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 13}")

    parser.add_argument("--class_num", type=int, default=1, help="Number of Total Categories")  # 60  6
    parser.add_argument("--seed", type=int, default=1024, help="random seed")
    parser.add_argument("--tile_size", type=int, default=608, help="image size")  # 300 416

    parser.add_argument("--syn_display_type", type=str, default='syn_texture',
                        help="syn_texture, syn_color, syn_mixed, syn_color0, syn_texture0, syn (match 0)")  # ######*********************change
    parser.add_argument("--syn_ratio", type=float, default=0.75,
                        help="ratio of synthetic data: 0.25, 0.5, 0.75, 1.0  0")  # ######*********************change

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
    return syn_args


if __name__ == "__main__":
    args = pwv.get_args()
    syn_args = get_part_syn_args()

    syn_args.data_xview_dir = syn_args.data_xview_dir.format(syn_args.class_num)

    '''
    IoU check by image name
    '''
    # score_thres = 0.3
    # iou_thres = 0.5
    # # # # image_name = '295_5.jpg'
    # # # # image_name = '80_2.jpg'
    # # image_name ='2518_2.jpg'
    # image_name ='1585_2.jpg' # iou=0.48
    # check_prd_gt_iou_xview_syn(dt, sr, image_name, score_thres, iou_thres)

    # score_thres = 0.3
    # iou_thres = 0.5
    # val_labels = pd.read_csv(os.path.join(syn_args.data_xview_dir, 'xviewval_lbl.txt'), header=None)
    # # display_type = ['syn_texture', 'syn_color', 'syn_mixed'] #, 'syn_texture0', 'syn_color0']
    # # syn_ratio = [0.25, 0.5, 0.75]
    # display_type = ['syn']
    # syn_ratio = [0]
    # for dt in display_type:
    #     for sr in syn_ratio:
    #         val_iou_map = {}
    #         for ix, vl in enumerate(val_labels.iloc[:, 0]):
    #             img_name = os.path.basename(vl).replace(TXT_SUFFIX, IMG_SUFFIX)
    #             iou_list = check_prd_gt_iou_xview_syn(dt, sr, img_name, score_thres, iou_thres)
    #             val_iou_map[ix] = iou_list
    #
    #         val_iou_path = os.path.join(args.txt_save_dir, 'val_result_iou_map')
    #         if not os.path.exists(val_iou_path):
    #             os.makedirs(val_iou_path)
    #         iou_json_file = os.path.join(val_iou_path, 'xViewval_{}_{}_iou.json'.format(dt, sr))  # topleft
    #         json.dump(val_iou_map, open(iou_json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)

    '''
    plot IOU distribution
    x-axis: IoU 
    y-axis: Number of Images
    '''
    # plot_val_results_iou_comp()

    '''
    val gt and prd results FP FN NMS
    '''
    # score_thres = 0.3
    # px_thres = 6
    # whr_thres = 4
    # iou_thres = 0.5
    # catid = 0
    # # display_type = ['syn_texture', 'syn_color', 'syn_mixed'] #, 'syn_texture0', 'syn_color0']
    # # syn_ratio = [0.25, 0.5, 0.75]
    # display_type = ['syn']
    # syn_ratio = [0]
    # for dt in display_type:
    #     for sr in syn_ratio:
    #         get_fp_fn_list_airplane(dt, sr, catid, iou_thres, score_thres, px_thres, whr_thres)

    '''
    plot val images with fp fn bbox
    '''
    # display_type = ['syn_texture', 'syn_color', 'syn_mixed'] #, 'syn_texture0', 'syn_color0']
    # syn_ratio = [0.25, 0.5, 0.75]
    # display_type = ['syn']
    # syn_ratio = [0]
    # for dt in display_type:
    #     for sr in syn_ratio:
    #         plot_val_img_with_fp_fn_bbox(dt, sr)

    '''
    statistic number of FP and number of FN
    see if the synthtetic data reduce the FP and FN 
    '''
    # draw_bar_compare_fp_fn_number_of_different_syn_ratio()

    # syn_ratios = [0.25] #, 0.5, 0.75]
    # for r in syn_ratios:
    #     draw_bar_compare_fp_fn_number_by_syn_ratio(r)

    '''
    look for the reduced fp and fn bbox
    '''
    # display_type = ['syn_texture', 'syn_color', 'syn_mixed']  # , 'syn_texture0', 'syn_color0']
    # syn_ratio = [0.25, 0.5, 0.75]
    # for dt in display_type:
    #     for sr in syn_ratio:
    #         look_for_reduced_fp_fn_in_image(dt, sr)

    '''
    plot val images with gt prd bbox
    '''
    # score_thres = 0.3
    # px_thres = 6
    # whr_thres = 4
    # catid = 0
    # display_type = ['syn_texture', 'syn_color', 'syn_mixed'] #, 'syn_texture0', 'syn_color0']
    # syn_ratio = [0.25, 0.5, 0.75]
    # for dt in display_type:
    #     for sr in syn_ratio:
    #       plot_val_img_with_gt_prd_bbox(dt, sr, catid, score_thres, px_thres, whr_thres)



