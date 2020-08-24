import numpy as np
import pandas as pd
import os
from utils.object_score_util.get_bbox_coords_from_annos_with_object_score import plot_img_with_bbx

if __name__ == '__main__':
    # sd = 17
    # # ap_list = [20, 40, 50]
    # # ehtypes = ['hard', 'easy']
    # model_ids = [4, 1, 5, 5, 5]
    # rare_classes = [1, 2, 3, 4, 5]


    # new_dir = '/media/lab/Yang/data/xView_YOLO/labels/608/1114_changed/'
    new_dir = '/media/lab/Yang/data/xView_YOLO/labels/608/1114_changed/unlabeled'
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    img_save_dir = '/media/lab/Yang/data/xView_YOLO/cat_samples/608/1_cls/image_with_bbox_indices/px23whr3_images_with_bbox_with_indices'

    # lbl_dir = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px23whr3_all_model/1114_2.txt'
    # lbl_dir = '/media/lab/Yang/data/xView_YOLO/labels/608/1114_orginal/1114_2.txt' # w*0.9 h*0.8
    lbl_dir = '/media/lab/Yang/data/xView_YOLO/labels/608/1114_original/unlabeled/1114_2.txt'
    # lbl_dir = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px23whr3/1114_2.txt'# w*0.9 h*0.68 _all_model
    img_dir = '/media/lab/Yang/data/xView_YOLO/images/608_1cls_rc_val/1114_2.jpg'
    ixes = [1,2,3,6,7,8,10]
    lbl = pd.read_csv(lbl_dir, header=None, sep=' ')
    for ix in ixes:
        lbl.loc[ix, 3] *= 0.9
        lbl.loc[ix, 4] *= 0.68
    #fixme replace the original one
    # lbl.to_csv(lbl_dir, sep=' ', header=False, index=False)
    # plot_img_with_bbx(img_dir, lbl_dir, img_save_dir, label_index=True)
    #fixme save the new one
    new_file = os.path.join(new_dir, os.path.basename(lbl_dir))
    lbl.to_csv(new_file, sep=' ', header=False, index=False)
    plot_img_with_bbx(img_dir, new_file, img_save_dir, label_index=True)

    # lbl_dir = '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px23whr3_all_model/1114_1.txt' # w*0.9 h*0.8
    # lbl_dir = '/media/lab/Yang/data/xView_YOLO/labels/608/1114_orginal/1114_1.txt' # w*0.9 h*0.8
    # lbl_dir = '/media/lab/Yang/data/xView_YOLO/labels/608/1114_original/unlabeled/1114_1.txt'
    # img_dir = '/media/lab/Yang/data/xView_YOLO/images/608_1cls_rc_train/1114_1.jpg'
    # ixes = [6, 10, 13]  # 10 is at the edge
    # lbl = pd.read_csv(lbl_dir, header=None, sep=' ')
    # for ix in ixes:
    #     lbl.loc[ix, 3] *= 0.9
    #     lbl.loc[ix, 4] *= 0.8
    #     if ix == 10:
    #         lbl.loc[ix, 2] -= 0.004
    # # lbl.to_csv(lbl_dir, sep=' ', header=False, index=False)
    # # plot_img_with_bbx(img_dir, lbl_dir, img_save_dir, label_index=True)
    # new_file = os.path.join(new_dir, os.path.basename(lbl_dir))
    # lbl.to_csv(new_file, sep=' ', header=False, index=False)
    # plot_img_with_bbx(img_dir, new_file, img_save_dir, label_index=True)

