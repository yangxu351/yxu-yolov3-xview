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


def plot_val_results_iou_comp_with_model_id(mid, comments=''):
    val_iou_path = os.path.join(args.txt_save_dir, 'val_result_iou_map', comments, 'model_{}'.format(mid))

    display_type = ['syn']
    syn_ratio = [0]

    syn0_iou_json_file = os.path.join(val_iou_path, 'xViewval_syn_0_iou.json')
    syn0_iou_map = json.load(open(syn0_iou_json_file))
    syn0_iou_list = []
    for v in syn0_iou_map.values():
        if len(v):
            syn0_iou_list.extend(v)

    colors = ['orange', 'm', 'g']
    display_type = ['syn_texture', 'syn_color', 'syn_mixed']  # , 'syn_texture0', 'syn_color0']
    syn_ratio = [0.25, 0.5, 0.75]
    # display_type = ['syn_texture']  # , 'syn_texture0', 'syn_color0']
    # syn_ratio = [0.25]
    fig, axs = plt.subplots(3, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ix, sr in enumerate(syn_ratio):
        for jx, dt in enumerate(display_type):
            iou_json_file = os.path.join(val_iou_path, 'xViewval_{}_{}_iou.json'.format(dt, sr))
            if not os.path.exists(iou_json_file):
                continue
            iou_map = json.load(open(iou_json_file))
            iou_list = []
            for v in iou_map.values():
                if len(v):
                    iou_list.extend(v)
            axs[ix, jx].hist(syn0_iou_list, histtype="bar", alpha=0.75, label='syn_0') #  bins=10, density=True,
            # if len(iou_list) < 10:
            #     axs[ix, jx].hist(iou_list,  bins=10,  histtype="bar", alpha=0.75, color=colors[jx],
            #                      label='{}_{}'.format(dt, sr))  # facecolor='g',
            # else:
            # axs[ix, jx].hist(iou_list, histtype="bar", density=True, alpha=0.75, color=colors[jx],
            #                  label='{}_{}'.format(dt, sr))
            axs[ix, jx].hist(iou_list, histtype="bar", alpha=0.75, color=colors[jx],
                             label='{}_{}'.format(dt, sr))
            axs[ix, jx].grid(True)
            axs[ix, jx].legend()
            axs[0, jx].set_title(dt)
            axs[2, jx].set_xlabel('IOU')
        axs[ix, 0].set_ylabel('Number of Images')
    fig.suptitle('Val Results IoU Comparison ' + comments + ' {}'.format(mid), fontsize=20)
    save_dir = os.path.join(args.txt_save_dir, 'val_result_iou_map', 'figures', comments)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig.savefig(os.path.join(save_dir, 'val_result_iou_comp_model_{}.png'.format(mid)))
    fig.show()


def get_tp_fn_list_airplane_with_model(dt, sr, comments=[], mid=0, catid=0, iou_thres=0.5, score_thres=0.3,
                                       px_thres=6, whr_thres=4, syn_cmt=''):
    ''' get TP FN of different 3d-models '''
    # print(os.path.join(syn_args.results_dir.format(syn_args.class_num, dt, sr), 'test_on*{}'.format(syn_cmt)))
    results_dir = glob.glob(os.path.join(syn_args.results_dir.format(syn_args.class_num, dt, sr), 'test_on*{}'.format(syn_cmt)))[0]
    result_json_file = os.path.join(results_dir, 'results_{}_{}.json'.format(sr, 'on_original_model'))
    result_allcat_list = json.load(open(result_json_file))
    result_list = []
    # #fixme filter, and rare_result_allcat_list contains rare_cat_ids, rare_img_id_list and object score larger than score_thres
    for ri in result_allcat_list:
        if ri['category_id'] == catid and ri['score'] >= score_thres:  # ri['image_id'] in rare_img_id_list and
            result_list.append(ri)
    print('len result_list', len(result_list))
    del result_allcat_list

    val_lbl_txt = 'xviewval_lbl_{}_with_model.txt'.format(comments[0])
    val_labels = pd.read_csv(os.path.join(syn_args.data_xview_dir, comments[0], val_lbl_txt), header=None)
    img_name_2_tp_list_maps = {}
    img_name_2_fn_list_maps = {}
    gt_cnt = 0
    for ix, vl in enumerate(val_labels.iloc[:, 0]):
        img_name = os.path.basename(vl).replace(TXT_SUFFIX, IMG_SUFFIX)
        img_name_2_tp_list_maps[img_name] = []
        img_name_2_fn_list_maps[img_name] = []
        good_gt_list = []
        if is_non_zero_file(vl):
            df_lbl = pd.read_csv(vl, header=None, delimiter=' ')
            df_lbl.iloc[:, 1:5] = df_lbl.iloc[:, 1:5] * syn_args.tile_size
            df_lbl.iloc[:, 1] = df_lbl.iloc[:, 1] - df_lbl.iloc[:, 3] / 2
            df_lbl.iloc[:, 2] = df_lbl.iloc[:, 2] - df_lbl.iloc[:, 4] / 2
            df_lbl.iloc[:, 3] = df_lbl.iloc[:, 1] + df_lbl.iloc[:, 3]
            df_lbl.iloc[:, 4] = df_lbl.iloc[:, 2] + df_lbl.iloc[:, 4]
            df_lbl = df_lbl.to_numpy()
            for dx in range(df_lbl.shape[0]):
                w, h = df_lbl[dx, 3] - df_lbl[dx, 1], df_lbl[dx, 4] - df_lbl[dx, 2]
                whr = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
                if whr <= whr_thres and w >= px_thres and h >= px_thres and df_lbl[dx, 5] == mid:
                    good_gt_list.append(df_lbl[dx, :])
        else:
            continue
        if not len(good_gt_list):
            continue
        gt_boxes = []
        if good_gt_list:
            good_gt_arr = np.array(good_gt_list)
            gt_boxes = good_gt_arr[:, 1:5]  # x1y1x2y2
            gt_classes = good_gt_arr[:, 0]
            gt_models = good_gt_arr[:, -1]
        # print('len of good gt ', len(good_gt_list))
        # gt_cnt += len(good_gt_list)

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

        dt_boxes = []
        if prd_lbl_list:
            prd_lbl_arr = np.array(prd_lbl_list)
            # print(prd_lbl_arr.shape)
            dt_scores = prd_lbl_arr[:, -1]  # [prd_lbl_arr[:, -1] >= score_thres]
            dt_boxes = prd_lbl_arr[dt_scores >= score_thres][:, 1:-1]
            dt_classes = prd_lbl_arr[dt_scores >= score_thres][:, 0]

        matches = []
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
        gt_cnt += len(gt_boxes)
        for i in range(len(gt_boxes)):
            if matches.shape[0] > 0 and matches[matches[:, 0] == i].shape[0] == 1:
                mt_i = matches[matches[:, 0] == i][0].tolist()
                # print(len(mt_i))
                # print('mt_i', mt_i.shape)
                c_box_iou = [gt_classes[i]]
                c_box_iou.extend(gt_boxes[i])
                c_box_iou.append(mt_i[-1])
                c_box_iou.append(gt_models[i]) # [cat_id, box[0:4], iou, model_id]
                img_name_2_tp_list_maps[img_name].append(c_box_iou)
            else:
                #fixme
                # unique matches at most has one match for each ground truth
                # 1. ground truth id deleted due to duplicate detections  --> FN
                # 2. matches.shape[0] == 0 --> no matches --> FN
                c_box_iou = [gt_classes[i]]
                c_box_iou.extend(gt_boxes[i])
                c_box_iou.append(0)
                c_box_iou.append(gt_models[i]) # [cat_id, box[0:4], iou, model_id]
                img_name_2_fn_list_maps[img_name].append(c_box_iou)
    print('ground truth count ', gt_cnt)
    tv = []
    for v in img_name_2_tp_list_maps.values():
        for vi in range(len(v)):
            tv.append(v[vi])
    fv = []
    for v in img_name_2_fn_list_maps.values():
        for vi in range(len(v)):
            fv.append(v[vi])
    print('tv ', len(tv), 'fv ', len(fv))

    save_dir = os.path.join(args.txt_save_dir,
                            'val_img_2_tp_fn_list', comments[0] + comments[1], '{}_{}{}/'.format(dt, sr, syn_cmt))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tp_json_file = os.path.join(save_dir,
                                'xViewval_{}_{}{}_img_2_tp_maps_model_{}.json'.format(dt, sr, syn_cmt, mid))  # topleft
    json.dump(img_name_2_tp_list_maps, open(tp_json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)

    fn_json_file = os.path.join(save_dir,
                                'xViewval_{}_{}{}_img_2_fn_maps_model_{}.json'.format(dt, sr, syn_cmt, mid))  # topleft
    json.dump(img_name_2_fn_list_maps, open(fn_json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)


def plot_val_img_with_tp_fn_bbox_with_model(dt, sr, comments='', mid=0, syn_cmt=''):
    tp_fn_list_dir = os.path.join(args.txt_save_dir,
                                  'val_img_2_tp_fn_list', comments, '{}_{}{}/'.format(dt, sr, syn_cmt))
    img_tp_fn_bbox_path = os.path.join(args.cat_sample_dir,
                                       'val_img_with_tp_fn_bbox', comments, '{}_{}{}/'.format(dt, sr, syn_cmt), 'model_{}'.format(mid))
    if not os.path.exists(img_tp_fn_bbox_path):
        os.makedirs(img_tp_fn_bbox_path)

    tp_maps = json.load(open(os.path.join(tp_fn_list_dir,
                                          'xViewval_{}_{}{}_img_2_tp_maps_model_{}.json'.format(dt, sr, syn_cmt, mid))))
    fn_maps = json.load(open(os.path.join(tp_fn_list_dir,
                                          'xViewval_{}_{}{}_img_2_fn_maps_model_{}.json'.format(dt, sr, syn_cmt, mid))))
    tp_color = (0, 255, 0)  # Green
    fn_color = (255, 0, 0)  # Blue
    tp_img_names = [k for k in tp_maps.keys()]
    fn_img_names = [k for k in fn_maps.keys()]
    img_names = tp_img_names + [k for k in fn_img_names if k not in tp_img_names]
    # print(len(img_names))
    for name in img_names:
        img = cv2.imread(os.path.join(args.images_save_dir, name))
        tp_list = tp_maps[name]
        fn_list = fn_maps[name]
        if not tp_list and not fn_list:
            continue
        for i in range(len(fn_list)):
            gr = fn_list[i]
            gr_bx = [int(x) for x in gr[1:-1]]
            img = cv2.rectangle(img, (gr_bx[0], gr_bx[1]), (gr_bx[2], gr_bx[3]), fn_color, 2)  # w1, h1, w2, h2
            # img = pwv.drawrect(img, (gr_bx[0], gr_bx[1]), (gr_bx[2], gr_bx[3]), fn_color, thickness=2, style='dotted')
            cv2.putText(img, text=str(int(gr[-1])), org=(gr_bx[2] - 8, gr_bx[3] - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 255))
        for px in range(len(tp_list)):
            pr = tp_list[px]
            pr_bx = [int(x) for x in pr[1:-1]]
            img = cv2.rectangle(img, (pr_bx[0], pr_bx[1]), (pr_bx[2], pr_bx[3]), tp_color, 2)
            # img = pwv.drawrect(img, (pr_bx[0], pr_bx[1]), (pr_bx[2], pr_bx[3]), tp_color, thickness=1, style='dotted')
            cv2.putText(img, text='{}'.format(int(pr[-1])), org=(pr_bx[0] + 5, pr_bx[1] + 8),  # [pr_bx[0], pr[-1]]
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 0))
        cv2.imwrite(os.path.join(img_tp_fn_bbox_path, name), img)


def draw_bar_compare_tp_fn_number_of_different_syn_ratio(comments='', mid=0):
    tp_fn_0_dir = os.path.join(args.txt_save_dir, 'val_img_2_tp_fn_list', comments[0] + '_with_model', comments[0])
    tp_0_file = json.load(open(os.path.join(tp_fn_0_dir, 'xViewval_{}_img_2_tp_maps_model_{}.json'.format(comments[0], mid))))
    fn_0_file = json.load(open(os.path.join(tp_fn_0_dir, 'xViewval_{}_img_2_fn_maps_model_{}.json'.format(comments[0], mid))))
    tp_0 = []
    for v in tp_0_file.values():
        for vi in range(len(v)):
            tp_0.append(v[vi])
    fn_0 = []
    for v in fn_0_file.values():
        for vi in range(len(v)):
            fn_0.append(v[vi])
    tp_0_num = len(tp_0)
    fn_0_num = len(fn_0)

    save_dir = os.path.join(args.txt_save_dir, 'val_img_2_tp_fn_list', 'figures', comments[1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    x = [1, 3, 5, 7, 9, 11]
    xlabels = ['TP seed=17',  'FN seed=17']
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    fig, ax = plt.subplots(1, 1)
    width = 0.3
    tp_fn_tx_path = os.path.join(args.txt_save_dir, 'val_img_2_tp_fn_list', comments[0] + '_with_model', comments[1])
    tp_tx_file = json.load(
        open(os.path.join(tp_fn_tx_path, 'xViewval_{}_img_2_tp_maps_model_{}.json'.format(comments[1], mid))))
    fn_tx_file = json.load(
        open(os.path.join(tp_fn_tx_path, 'xViewval_{}_img_2_fn_maps_model_{}.json'.format(comments[1],mid))))
    tp_tx = []
    for v in tp_tx_file.values():
        for vi in range(len(v)):
            tp_tx.append(v[vi])
    fn_tx = []
    for v in fn_tx_file.values():
        for vi in range(len(v)):
            fn_tx.append(v[vi])
    tp_tx_num = len(tp_tx)
    fn_tx_num = len(fn_tx)

        # tp_fn_clr_path = os.path.join(args.txt_save_dir, 'val_img_2_tp_fn_list', comments,
        #                               '{}_{}'.format('syn_color', r))
        # if not os.path.exists(tp_fn_clr_path):
        #     os.makedirs(tp_fn_clr_path)
        # tp_clr_file = json.load(
        #     open(os.path.join(tp_fn_clr_path, 'xViewval_{}_{}_img_2_tp_maps_model_{}.json'.format('syn_color', r, mid))))
        # fn_clr_file = json.load(
        #     open(os.path.join(tp_fn_clr_path, 'xViewval_{}_{}_img_2_fn_maps_model_{}.json'.format('syn_color', r, mid))))
        # tp_clr_num = len([k for k in tp_clr_file.keys() if tp_clr_file.get(k)])
        # fn_clr_num = len([k for k in fn_clr_file.keys() if fn_clr_file.get(k)])
        #
        # tp_fn_mx_path = os.path.join(args.txt_save_dir, 'val_img_2_tp_fn_list', comments,
        #                              '{}_{}'.format('syn_mixed', r))
        # if not os.path.exists(tp_fn_mx_path):
        #     os.makedirs(tp_fn_mx_path)
        # tp_mx_file = json.load(
        #     open(os.path.join(tp_fn_mx_path, 'xViewval_{}_{}_img_2_tp_maps_model_{}.json'.format('syn_mixed', r, mid))))
        # fn_mx_file = json.load(
        #     open(os.path.join(tp_fn_mx_path, 'xViewval_{}_{}_img_2_fn_maps_model_{}.json'.format('syn_mixed', r, mid))))
        # tp_mx_num = len([k for k in tp_mx_file.keys() if tp_mx_file.get(k)])
        # fn_mx_num = len([k for k in fn_mx_file.keys() if fn_mx_file.get(k)])

    rects_syn_0 = ax.bar([x[0] - width, x[1] - width], [tp_0_num, fn_0_num], width, label=comments[0])
    autolabel(ax, rects_syn_0, x, xlabels, [tp_0_num, fn_0_num], rotation=0)

        # rects_syn_clr = ax.bar([x[ix], x[ix + 3]], [tp_clr_num, fn_clr_num], width,
        #                        label='syn_color_ratio={}'.format(r))  # , label=labels
        # autolabel(ax, rects_syn_clr, x, xlabels, [tp_clr_num, fn_clr_num], rotation=0)

    rects_syn_tx = ax.bar([x[0] + width, x[1] + width], [tp_tx_num, fn_tx_num], width,
                          label=comments[1])  # , label=labels
    autolabel(ax, rects_syn_tx, x, xlabels, [tp_tx_num, fn_tx_num], rotation=0)

        # rects_syn_mx = ax.bar([x[ix] + 2 * width, x[ix + 3] + 2 * width], [tp_mx_num, fn_mx_num], width,
        #                       label='syn_mixed_ratio={}'.format(r))  # , label=labels
        # autolabel(ax, rects_syn_mx, x, xlabels, [tp_mx_num, fn_mx_num], rotation=0)

    ax.legend()
    ylabel = "Number"
    plt.title('{} 3-d Model {}'.format(comments[1], mid), literal_eval(syn_args.font2))
    plt.ylabel(ylabel, literal_eval(syn_args.font2))
    plt.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'cmp_tp_fn_{}_vs_{}_model_{}.jpg'.format(comments[0], comments[1], mid)))
    plt.show()

def draw_bar_compare_tp_fn_number_of_different_comments(comments='', mid=0, xview_cmt=''):
    tp_fn_0_dir = os.path.join(args.txt_save_dir, 'val_img_2_tp_fn_list', comments[0] + '_with_model', comments[0])
    tp_0_file = json.load(open(os.path.join(tp_fn_0_dir, 'xViewval_{}_img_2_tp_maps_model_{}.json'.format(comments[0], mid))))
    fn_0_file = json.load(open(os.path.join(tp_fn_0_dir, 'xViewval_{}_img_2_fn_maps_model_{}.json'.format(comments[0], mid))))
    tp_0 = []
    for v in tp_0_file.values():
        for vi in range(len(v)):
            tp_0.append(v[vi])
    fn_0 = []
    for v in fn_0_file.values():
        for vi in range(len(v)):
            fn_0.append(v[vi])
    tp_0_num = len(tp_0)
    fn_0_num = len(fn_0)

    save_dir = os.path.join(args.txt_save_dir, 'val_img_2_tp_fn_list', 'figures', comments[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    x = [1, 3, 5, 7, 9, 11]
    xlabels = ['TP seed=17',  'FN seed=17']
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    fig, ax = plt.subplots(1, 1)
    width = 0.3
    tp_fn_tx_path = os.path.join(args.txt_save_dir, 'val_img_2_tp_fn_list', comments[0] + '_with_model', comments[1] + xview_cmt)
    tp_tx_file = json.load(
        open(os.path.join(tp_fn_tx_path, 'xViewval_{}{}_img_2_tp_maps_model_{}.json'.format(comments[1], xview_cmt, mid))))
    fn_tx_file = json.load(
        open(os.path.join(tp_fn_tx_path, 'xViewval_{}{}_img_2_fn_maps_model_{}.json'.format(comments[1], xview_cmt, mid))))
    tp_tx = []
    for v in tp_tx_file.values():
        for vi in range(len(v)):
            tp_tx.append(v[vi])
    fn_tx = []
    for v in fn_tx_file.values():
        for vi in range(len(v)):
            fn_tx.append(v[vi])
    tp_tx_num = len(tp_tx)
    fn_tx_num = len(fn_tx)

    # tp_fn_cl_path = os.path.join(args.txt_save_dir, 'val_img_2_tp_fn_list', comments[0] + '_with_model', comments[2])
    # tp_cl_file = json.load(
    #     open(os.path.join(tp_fn_cl_path, 'xViewval_{}_img_2_tp_maps_model_{}.json'.format(comments[2], mid))))
    # fn_cl_file = json.load(
    #     open(os.path.join(tp_fn_cl_path, 'xViewval_{}_img_2_fn_maps_model_{}.json'.format(comments[2],mid))))
    # tp_cl = []
    # for v in tp_cl_file.values():
    #     for vi in range(len(v)):
    #         tp_cl.append(v[vi])
    # fn_cl = []
    # for v in fn_cl_file.values():
    #     for vi in range(len(v)):
    #         fn_cl.append(v[vi])
    # tp_cl_num = len(tp_cl)
    # fn_cl_num = len(fn_cl)

    rects_syn_0 = ax.bar([x[0] - width, x[1] - width], [tp_0_num, fn_0_num], width, label=comments[0])
    autolabel(ax, rects_syn_0, x, xlabels, [tp_0_num, fn_0_num], rotation=0)

    rects_syn_tx = ax.bar([x[0], x[1]], [tp_tx_num, fn_tx_num], width, label=comments[1] + xview_cmt)  # , label=labels
    autolabel(ax, rects_syn_tx, x, xlabels, [tp_tx_num, fn_tx_num], rotation=0)

    # rects_syn_clr = ax.bar([x[0] + width, x[1] + width], [tp_cl_num, fn_cl_num], width, label=comments[2] + xview_cmt)  # , label=labels
    # autolabel(ax, rects_syn_clr, x, xlabels, [tp_cl_num, fn_cl_num], rotation=0)

    ax.legend()
    ylabel = "Number"
    plt.title('{} Algorithms on 3-d Model {}'.format(len(comments), mid), literal_eval(syn_args.font2))
    plt.ylabel(ylabel, literal_eval(syn_args.font2))
    plt.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.grid()
    # plt.savefig(os.path.join(save_dir, 'cmp_tp_fn_{}_vs_{}_{}_model_{}.jpg'.format(comments[0], comments[1], comments[2], mid)))
    plt.savefig(os.path.join(save_dir, 'cmp_tp_fn_{}_vs_{}_model_{}.jpg'.format(comments[0], comments[1]+xview_cmt, mid)))
    plt.show()


def draw_bar_compare_tp_number_of_different_models(comments=[], mids=[], xview_cmt=['']):
    save_dir = os.path.join(args.txt_save_dir, 'val_img_2_tp_fn_list', 'figures', comments[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # xlabels = ['Model0', 'Model1', 'Model2', 'Unlabeled']
    # x = np.array([1, 3, 5, 7])
    xlabels = ['Model0', 'Model1', 'Model2', 'Model3', 'Model4', 'Model5']
    x = np.arange(1, 1+len(mids)*2, 2)

    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    fig, (axs1, axs2) = plt.subplots(2, 1) #,  sharex=True, sharey=True
    width = 0.3
    for cix, cmt in enumerate(comments):
        tp_fn_tx_path = os.path.join(args.txt_save_dir, 'val_img_2_tp_fn_list',  comments[0] + '_with_model', cmt + xview_cmt[cix])
        tp_num_arr = np.zeros((len(mids)), dtype=np.int)
        fn_num_arr = np.zeros((len(mids)), dtype=np.int)
        for mid in mids:
            tp_tx_file = json.load(
                open(os.path.join(tp_fn_tx_path, 'xViewval_{}{}_img_2_tp_maps_model_{}.json'.format(cmt, xview_cmt[cix], mid))))
            fn_tx_file = json.load(
                open(os.path.join(tp_fn_tx_path, 'xViewval_{}{}_img_2_fn_maps_model_{}.json'.format(cmt, xview_cmt[cix], mid))))
            for v in tp_tx_file.values():
                tp_num_arr[mid] += len(v)
            for v in fn_tx_file.values():
                fn_num_arr[mid] += len(v)
        rects_syn_tp = axs1.bar(x + cix*width, tp_num_arr, width, label=cmt + xview_cmt[cix])  # , label=labels
        autolabel(axs1, rects_syn_tp, x + cix*width, xlabels, tp_num_arr, rotation=0)

        rects_syn_fn = axs2.bar(x + cix*width, fn_num_arr, width, label=cmt + xview_cmt[cix])  # , label=labels
        autolabel(axs2, rects_syn_fn, x + cix*width, xlabels, fn_num_arr, rotation=0)

    axs1.legend()
    axs2.legend()
    axs1.grid(True)
    axs2.grid(True)
    axs1.set_xlabel('TP', literal_eval(syn_args.font2))
    axs2.set_xlabel('FN', literal_eval(syn_args.font2))
    axs1.set_ylabel("Number", literal_eval(syn_args.font2))
    axs2.set_ylabel("Number", literal_eval(syn_args.font2))
    fig.suptitle('{} Algorithms of TP & RN on {} Models'.format(len(comments), len(mids)), fontsize=18)
    if xview_cmt:
        jpg_name = 'cmp_tp_fn_of_{}_algorithms_{}_*{}_on_{}_models.jpg'.format(len(comments), comments[1].split('_texture')[0], xview_cmt, len(mids))
    else:
        jpg_name = 'cmp_tp_fn_of_{}_algorithms_{}_*_on_{}_models.jpg'.format(len(comments), comments[1].split('_texture')[0], len(mids))
    plt.savefig(os.path.join(save_dir, jpg_name))
    plt.show()


def statistic_model_number(type='validation', comments='px6whr4_ng0_seed17'):

    # comments = '38bbox_giou0_with_model'
    if type == 'validation':
        val_lbl_file = '/media/lab/Yang/code/yolov3/data_xview/1_cls/{}/xviewval_lbl_{}_with_model.txt'.format(comments, comments)
        json_name = 'val_model_num_maps.json'
        png_name = 'val_number_3d-model.jpg'
    else:
        val_lbl_file = '/media/lab/Yang/code/yolov3/data_xview/1_cls/{}/xviewtrain_lbl_{}_with_model.txt'.format(comments, comments)
        json_name = 'trn_model_num_maps.json'
        png_name = 'trn_number_3d-model.jpg'
    df_val = pd.read_csv(val_lbl_file, header=None)
    Num = {}

    for f in df_val.loc[:, 0]:
        if not is_non_zero_file(f):
            continue
        df_lbl = pd.read_csv(f, header=None, sep=' ')
        for m in df_lbl.loc[:, 5]:
            if m not in Num.keys():
                Num[m] = 1
            else:
                Num[m] += 1
    json_dir = os.path.join(args.txt_save_dir, 'val_result_iou_map', comments)
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    json.dump(Num, open(os.path.join(json_dir, json_name),
                        'w'), ensure_ascii=False, indent=2, cls=MyEncoder)

    save_dir = os.path.join(args.txt_save_dir, 'val_result_iou_map', 'figures', comments)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    fig, ax = plt.subplots(1, 1)
    width = 0.35
    x = [k for k in Num.keys()]
    ylist = [v for v in Num.values()]
    rects = ax.bar(np.array(x), ylist, width)  # , label=labels
    autolabel(ax, rects, x, x, ylist, rotation=0)
    ylabel = 'Number of Bbox'
    xlabel = "Model ID"
    plt.title('Model Numbers in {} Dataset'.format(type), literal_eval(syn_args.font2))
    plt.ylabel(ylabel, literal_eval(syn_args.font2))
    plt.xlabel(xlabel, literal_eval(syn_args.font2))
    plt.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.grid()
    plt.savefig(os.path.join(save_dir, png_name))
    plt.show()


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
    syn_args.data_xview_dir = syn_args.data_xview_dir.format(syn_args.class_num)
    return syn_args


if __name__ == "__main__":
    args = pwv.get_args()
    syn_args = get_part_syn_args()

    '''
    IoU histograms
    '''
    # score_thres = 0.3
    # iou_thres = 0.5
    # # val_labels = pd.read_csv(os.path.join(syn_args.data_xview_dir, 'xviewval_lbl.txt'), header=None)
    # val_labels = pd.read_csv(os.path.join(syn_args.data_xview_dir, 'xviewval_lbl_with_model.txt'), header=None)
    # # display_type = ['syn_color', 'syn_texture', 'syn_mixed'] # , 'syn_texture0', 'syn_color0']
    # # syn_ratio = [0.25, 0.5, 0.75]
    # display_type = ['syn']
    # syn_ratio = [0]
    # # comments = ''
    # # comments = '38bbox_giou0'
    # comments = '38bbox_giou0_with_model'
    # model_ids = [0, 1, 2]
    # for mid in model_ids:
    #     for dt in display_type:
    #         for sr in syn_ratio:
    #             val_iou_map = {}
    #             for ix, vl in enumerate(val_labels.iloc[:, 0]):
    #                 txt_path = vl.split(os.path.basename(vl))[0]
    #                 img_name = os.path.basename(vl).replace(TXT_SUFFIX, IMG_SUFFIX)
    #                 iou_list = axs.check_prd_gt_iou_xview_syn(dt, sr, img_name, comments, txt_path, score_thres, iou_thres, mid)
    #                 val_iou_map[ix] = iou_list
    #
    #             val_iou_path = os.path.join(args.txt_save_dir, 'val_result_iou_map', comments, 'model_{}'.format(mid))
    #             if not os.path.exists(val_iou_path):
    #                 os.makedirs(val_iou_path)
    #             iou_json_file = os.path.join(val_iou_path, 'xViewval_{}_{}_iou.json'.format(dt, sr))  # topleft
    #             json.dump(val_iou_map, open(iou_json_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)

    '''
    plot IOU distribution
    x-axis: IoU 
    y-axis: Number of Images
    '''
    # comments = ''
    # comments = '38bbox_giou0'
    # comments = '38bbox_giou0_with_model'
    # model_ids = [0, 1, 2]
    # # model_ids = [2]
    # for mid in model_ids:
    #     plot_val_results_iou_comp_with_model_id(mid, comments)

    '''
    statistic model Number 
    '''
    # type = 'training'
    # type = 'validation'
    # # comments='px6whr4_ng0_seed17'
    # # comments='px20whr4_seed17'
    # # comments='px23whr4_seed17'
    # comments='px23whr3_seed17'
    # statistic_model_number(type, comments)

    '''
    val gt and prd results TP FN NMS
    '''
    # # # px_thres = 6
    # # # # comments = ['px6whr4_ng0_seed17', '_with_model']
    # # # # display_type = ['px6whr4_ng0']
    # # px_thres = 20
    # # comments = ['px20whr4_seed17', '_with_model']
    # # display_type = ['px20whr4']
    # px_thres = 23
    # whr_thres = 4
    # comments = ['px23whr4_seed17', '_with_model']
    # display_type = ['px23whr4']
    # px_thres = 23
    # whr_thres = 3
    # comments = ['px23whr3_seed17', '_with_model']
    # display_type = ['px23whr3']
    #
    # syn_ratio = ['seed17']
    # score_thres = 0.3
    # iou_thres = 0.5
    # catid = 0
    # # model_ids = [0, 1, 2, 3]
    # model_ids = [0, 1, 2, 3, 4, 5]
    # for mid in model_ids:
    #     for dt in display_type:
    #         for sr in syn_ratio:
    #             get_tp_fn_list_airplane_with_model(dt, sr, comments, mid, catid, iou_thres, score_thres, px_thres, whr_thres)
    #             plot_val_img_with_tp_fn_bbox_with_model(dt, sr, comments[0] + comments[1], mid)


    # comments = ['px6whr4_ng0_seed17', '_with_model']
    # px_thres = 6
    # display_types = ['syn_xview_bkg_certain_models_texture', 'syn_xview_bkg_certain_models_color',
    #                  'syn_xview_bkg_certain_models_mixed'] #
    # syn_cmt = ''
    # # comments = ['px20whr4_seed17', '_with_model']
    # # px_thres = 20
    # # display_types = ['syn_xview_bkg_px20whr4_certain_models_texture', 'syn_xview_bkg_px20whr4_certain_models_color',
    # #                  'syn_xview_bkg_px20whr4_certain_models_mixed']
    # syn_cmt = ''
    # px_thres = 23
    # comments = ['px23whr4_seed17', '_with_model']
    # display_types = ['syn_xview_bkg_px23whr4_scale_models_texture',
    #                 'syn_xview_bkg_px23whr4_scale_models_color',
    #                 'syn_xview_bkg_px23whr4_scale_models_mixed']
    #
    # sr = 'seed17'
    # score_thres = 0.3
    # whr_thres = 4
    # iou_thres = 0.5
    # catid = 0
    # model_ids = [0, 1, 2, 3]
    # for mid in model_ids:
    #     for dt in display_types:
    #         get_tp_fn_list_airplane_with_model(dt, sr, comments, mid, catid, iou_thres, score_thres, px_thres, whr_thres, syn_cmt)
    #         plot_val_img_with_tp_fn_bbox_with_model(dt, sr, comments[0] + comments[1], mid, syn_cmt)



    # # # px_thres = 6 ##*******
    # # # whr_thres = 4
    # # # comments = ['px6whr4_ng0_seed17', '_with_model']
    # # # display_types = ['xview_syn_xview_bkg_texture']
    # # # syn_cmt = '_1xSyn'
    # # # comments = ['px6whr4_ng0_seed17', '_with_model']
    # # # display_types = ['xview_syn_xview_bkg_texture', 'xview_syn_xview_bkg_color',
    # # #                  'xview_syn_xview_bkg_mixed']
    # # # syn_cmt = '_1xSyn'
    # # px_thres = 6
    # # # whr_thres = 4
    # # comments = ['px6whr4_ng0_seed17', '_with_model']
    # # display_types = ['xview_syn_xview_bkg_certain_models_texture', 'xview_syn_xview_bkg_certain_models_color',
    # #                  'xview_syn_xview_bkg_certain_models_mixed']
    # # syn_cmt = '_1xSyn'
    # px_thres = 6
    # # whr_thres = 4
    # comments = ['px6whr4_ng0_seed17', '_with_model']
    # display_types = ['xview_syn_xview_bkg_certain_models_texture']
    # # syn_cmt = '_1xSyn'
    # syn_cmt = '_2xSyn'
    # # # px_thres = 20 ##*******
    # # # whr_thres = 4
    # # # comments = ['px20whr4_seed17', '_with_model']
    # # # display_types = ['xview_syn_xview_bkg_px20whr4_certain_models_texture', 'xview_syn_xview_bkg_px20whr4_certain_models_color',
    # # #                  'xview_syn_xview_bkg_px20whr4_certain_models_mixed']
    # # # syn_cmt = '_1xSyn'

    # # px_thres = 23 ##*******
    # # whr_thres = 4
    # # comments = ['px23whr4_seed17', '_with_model']
    # # display_types = ['xview_syn_xview_bkg_px23whr4_scale_models_texture', 'xview_syn_xview_bkg_px23whr4_scale_models_color',
    # #                  'xview_syn_xview_bkg_px23whr4_scale_models_mixed']
    # # syn_cmt = '_1xSyn'
    # # px_thres = 23 ##*******
    # # whr_thres = 4
    # # comments = ['px23whr4_seed17', '_with_model']
    # # display_types = ['xview_syn_xview_bkg_px23whr4_small_models_color',
    # #                  'xview_syn_xview_bkg_px23whr4_small_models_mixed']
    # syn_cmt = '_1xSyn'
    # px_thres = 23 ##*******
    # whr_thres = 3
    # comments = ['px23whr3_seed17', '_with_model']
    # display_types = ['xview_syn_xview_bkg_px23whr3_6groups_models_color',
    #                  'xview_syn_xview_bkg_px23whr3_6groups_models_mixed']
    # syn_cmt = '_1xSyn'
    # score_thres = 0.3
    # iou_thres = 0.5
    # catid = 0
    # # model_ids = [0, 1, 2, 3]
    # model_ids = [0, 1, 2, 3, 4, 5]
    # sr = 'seed17'
    # for mid in model_ids:
    #     for dt in display_types:
    #         get_tp_fn_list_airplane_with_model(dt, sr, comments, mid, catid, iou_thres, score_thres, px_thres, whr_thres, syn_cmt)
    #         plot_val_img_with_tp_fn_bbox_with_model(dt, sr, comments[0] + comments[1], mid, syn_cmt)


    '''
    TP and FN
    compare
    '''
    # model_ids = [0, 1, 2, 3]
    # comments = ['px6whr4_ng0_seed17', 'xview_syn_xview_bkg_texture_seed17']
    # # comments = ['px6whr4_ng0_seed17', 'xview_syn_xview_bkg_certain_models_texture_seed17_1xSyn']
    # for mid in model_ids:
    #     draw_bar_compare_tp_fn_number_of_different_syn_ratio(comments, mid)

    # model_ids = [0, 1, 2, 3]
    # comments = ['px6whr4_ng0_seed17', 'syn_xview_bkg_texture_seed17', 'syn_xview_bkg_certain_models_texture_seed17_1xSyn']
    # for mid in model_ids:
    #     draw_bar_compare_tp_fn_number_of_different_comments(comments, mid)

    # model_ids = [0, 1, 2, 3]
    # comments = ['px6whr4_ng0_seed17', 'xview_syn_xview_bkg_texture_seed17',
    # 'xview_syn_xview_bkg_certain_models_texture_seed17_1xSyn']
    # for mid in model_ids:
    #     draw_bar_compare_tp_fn_number_of_different_comments(comments, mid)

    # model_ids = [0, 1, 2, 3]
    # comments = ['px6whr4_ng0_seed17', 'syn_xview_bkg_certain_models_texture_seed17', 'syn_xview_bkg_certain_models_color_seed17']
    # for mid in model_ids:
    #     draw_bar_compare_tp_fn_number_of_different_comments(comments, mid)

    # model_ids = [0, 1, 2, 3]
    # comments = ['px20whr4_seed17', 'syn_xview_bkg_px20whr4_certain_models_texture_seed17',
    #             'syn_xview_bkg_px20whr4_certain_models_color_seed17',
    #             'syn_xview_bkg_px20whr4_certain_models_mixed_seed17']
    # for mid in model_ids:
    #     draw_bar_compare_tp_fn_number_of_different_comments(comments, mid)

    # model_ids = [0, 1, 2, 3]
    # comments = ['px20whr4_seed17', 'xview_syn_xview_bkg_px20whr4_certain_models_texture_seed17']
    # xview_cmt = '_1xSyn'
    # for mid in model_ids:
    #     draw_bar_compare_tp_fn_number_of_different_comments(comments, mid, xview_cmt)

    '''
    TP and FN
    compare all
    '''
    # comments = ['px6whr4_ng0_seed17',
    #             'xview_syn_xview_bkg_texture_seed17',
    #             'xview_syn_xview_bkg_color_seed17',
    #             'xview_syn_xview_bkg_mixed_seed17']
    # xview_cmt = ['', '_1xSyn', '_1xSyn', '_1xSyn']
    # comments = ['px6whr4_ng0_seed17',
    #             'xview_syn_xview_bkg_texture_seed17',
    #             'xview_syn_xview_bkg_certain_models_texture_seed17']
    # xview_cmt = ['', '_1xSyn', '_1xSyn']
    # comments = ['px6whr4_ng0_seed17',
    #             'syn_xview_bkg_certain_models_texture_seed17',
    #             'syn_xview_bkg_certain_models_color_seed17']
    # , 'syn_xview_bkg_certain_models_mixed_seed17'
    # xview_cmt = ['', '', '']
    # comments = ['px6whr4_ng0_seed17',
    #             'xview_syn_xview_bkg_certain_models_texture_seed17'
    #             ,'xview_syn_xview_bkg_certain_models_texture_seed17']
    # # , 'xview_syn_xview_bkg_certain_models_mixed_seed17'
    # xview_cmt = ['', '_1xSyn', '_2xSyn']
    # comments = ['px20whr4_seed17',
    #             'syn_xview_bkg_px20whr4_certain_models_texture_seed17',
    #             'syn_xview_bkg_px20whr4_certain_models_color_seed17',
    #             'syn_xview_bkg_px20whr4_certain_models_mixed_seed17']
    # xview_cmt = ['', '', '', '']
    # comments = ['px20whr4_seed17',
    #             'xview_syn_xview_bkg_px20whr4_certain_models_texture_seed17',
    #             'xview_syn_xview_bkg_px20whr4_certain_models_color_seed17',
    #             'xview_syn_xview_bkg_px20whr4_certain_models_mixed_seed17']
    # xview_cmt = ['', '_1xSyn', '_1xSyn', '_1xSyn']
    # comments = ['px23whr4_seed17',
    #             'syn_xview_bkg_px23whr4_scale_models_texture_seed17',
    #             'syn_xview_bkg_px23whr4_scale_models_color_seed17',
    #             'syn_xview_bkg_px23whr4_scale_models_mixed_seed17']
    # xview_cmt = ['', '', '', '']
    # comments = ['px23whr4_seed17',
    #             'xview_syn_xview_bkg_px23whr4_scale_models_texture_seed17',
    #             'xview_syn_xview_bkg_px23whr4_scale_models_color_seed17',
    #             'xview_syn_xview_bkg_px23whr4_scale_models_mixed_seed17']
    # xview_cmt = ['', '_1xSyn', '_1xSyn', '_1xSyn']
    # comments = ['px23whr4_seed17',
    #             'xview_syn_xview_bkg_px23whr4_small_models_color_seed17',
    #             'xview_syn_xview_bkg_px23whr4_small_models_mixed_seed17']
    # xview_cmt = ['', '_1xSyn', '_1xSyn']
    # comments = ['px23whr3_seed17',
    #             'xview_syn_xview_bkg_px23whr3_6groups_models_color_seed17',
    #             'xview_syn_xview_bkg_px23whr3_6groups_models_mixed_seed17']
    # xview_cmt = ['', '_1xSyn', '_1xSyn']
    # model_ids = [0, 1, 2, 3, 4, 5]
    # # model_ids = [0, 1, 2, 3]
    # draw_bar_compare_tp_number_of_different_models(comments, model_ids, xview_cmt)


    '''
    Compute mAP of different models
    '''
    # whr_thres = 4
    # px_thres = 6
    #
    # gt_files = os.path.join(syn_args.data_xview_dir, 'xviewval_lbl_with_model.txt')
    # df_gts = pd.read_csv(gt_files, header=None)
    # for vl in df_gts:
    #     if is_non_zero_file(vl):
    #         df_lbl =  pd.read_csv(vl, header=None, delimiter=' ')
    #         df_lbl.iloc[:, 1:5] = df_lbl.iloc[:, 1:5] * syn_args.tile_size
    #         df_lbl.iloc[:, 1] = df_lbl.iloc[:, 1] - df_lbl.iloc[:, 3] / 2
    #         df_lbl.iloc[:, 2] = df_lbl.iloc[:, 2] - df_lbl.iloc[:, 4] / 2
    #         df_lbl.iloc[:, 3] = df_lbl.iloc[:, 1] + df_lbl.iloc[:, 3]
    #         df_lbl.iloc[:, 4] = df_lbl.iloc[:, 2] + df_lbl.iloc[:, 4]
    #         df_lbl = df_lbl.to_numpy()
    #         for dx in range(df_lbl.shape[0]):
    #             w, h = df_lbl[dx, 3] - df_lbl[dx, 1], df_lbl[dx, 4] - df_lbl[dx, 2]
    #             whr = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
    #             if whr <= whr_thres and w >= px_thres and h >= px_thres and df_lbl[dx, 5] == mid:
    #                 good_gt_list.append(df_lbl[dx, :])
    #
    # model_ids = [0, 1, 2]
    # for mid in model_ids:
    # import torch
    # a = torch.load('/media/lab/Yang/code/yolov3/weights/1_cls/xview_syn_xview_bkg_px23whr4_scale_models_texture_seed17/2020-04-18_10.07_hgiou1_seed17_1xSyn/backup160.pt')
    # b = torch.load('/media/lab/Yang/code/yolov3/weights/1_cls/xview_syn_xview_bkg_px23whr4_scale_models_texture_seed17/2020-04-19_07.31_hgiou1_seed17_1xSyn/backup160.pt')
    # print(a==b)



