import glob
import numpy as np
import os
import sys
sys.path.append('/data/users/yang/code/yxu-yolov3-xview')
from utils.object_score_util import get_bbox_coords_from_annos_with_object_score as gbc
import pandas as pd
import cv2
import shutil
import argparse

IMG_FORMAT = '.png'
TXT_FORMAT = '.txt'

def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def convert_norm(size, box):
    '''
    https://blog.csdn.net/xuanlang39/article/details/88642010
    :param size:  w h
    :param box: y1 x1 y2 x2
    :return: xc yc w h  (relative values)
    '''
    dh = 1. / (size[1])  # h--1--y
    dw = 1. / (size[0])  # w--0--x

    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0  # (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


def resize_crop_windturbine(px_thres = 10):
    wt_dir = syn_args.syn_data_dir
    wt_images = np.sort(glob.glob(os.path.join(wt_dir, '*.jpg')))
    wt_lbls = np.sort(glob.glob(os.path.join(wt_dir, '*.txt')))
    save_img_dir = syn_args.syn_img_dir
    if not os.path.exists(save_img_dir):
        os.mkdir(save_img_dir)
    save_lbl_dir = syn_args.syn_annos_dir
    if not os.path.exists(save_lbl_dir):
        os.mkdir(save_lbl_dir)
    w = syn_args.tile_size
    h = syn_args.tile_size

    for i in range(len(wt_images)):
        img = cv2.imread(wt_images[i])
        name = os.path.basename(wt_images[i])
        # if name != 'naip_915_CA_WND.jpg':
        #     continue
        height, width = img.shape[:2]
        rnum = np.ceil(height/h).astype(np.int)
        cnum = np.ceil(width/w).astype(np.int)
        # print('label file', wt_lbls[i])
        tl_w, tl_h = 0, 0
        br_w, br_h = width, height
        c_w, c_h = br_w-w, br_h-h

        for r in range(rnum):
            for c in range(cnum):
                wmin = c*w
                wmax = (c+1)*w
                hmin = r*h
                hmax = (r+1)*h
                if hmax >= height and wmax >= width:
                    hmin = height - h
                    wmin = width - w
                    chip = img[hmin : height, wmin : width, :3]
                elif hmax >= height and wmax < width:
                    hmin = height - h
                    chip = img[hmin : height, wmin : wmax, :3]
                elif hmax < height and wmax >= width:
                    wmin = width - w
                    chip = img[hmin : hmax, wmin : width, :3]
                else:
                    chip = img[hmin : hmax, wmin : wmax, :3]
                cv2.imwrite(os.path.join(save_img_dir, name.split('.')[0] + '_i{}j{}.jpg'.format(r, c)), chip)

        if not is_non_zero_file(wt_lbls[i]):
            shutil.copy(wt_lbls[i], os.path.join(save_lbl_dir, name.replace('.jpg', '.txt')))
        else:
            lbl = pd.read_csv(wt_lbls[i], header=None, sep=' ').to_numpy()
            b0_list = []
            b1_list = []
            b2_list = []
            b3_list = []
            bboxes = lbl[:, 1:]# xc, yc, w, h
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * width
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * height
            bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2]/2
            bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3]/2
            bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
            bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
            for ti in range(bboxes.shape[0]):
                bbox = bboxes[ti, :] # x1, y1, x2, y2
                b0 = np.clip(bbox, [tl_w, tl_h, tl_w, tl_h], [w-1, h-1, w-1, h-1])
                b1 = bbox.copy()
                # print('b1', b1.shape)
                b1[[0, 2]] = b1[[0, 2]] - c_w
                b1 = np.clip(b1, [tl_w, tl_h, tl_w, tl_h], [w-1, h-1, w-1, h-1])
                b2 = bbox.copy()
                b2[[1, 3]] = b2[[1, 3]] - c_h
                b2 = np.clip(b2, [tl_w, tl_h, tl_w, tl_h], [w-1, h-1, w-1, h-1])
                b3 = bbox.copy()
                b3[[0, 2]] = b3[[0, 2]] - c_w
                b3[[1, 3]] = b3[[1, 3]] - c_h
                b3 = np.clip(b3, [tl_w, tl_h, tl_w, tl_h], [w-1, h-1, w-1, h-1])
                if (b0[2] < w-1 and b0[3] < h-1) or (b0[2] == w-1 and  b0[2]-b0[0] > px_thres) or (b0[3] == h-1 and b0[3]-b0[1] > px_thres):
                    b0_list.append(convert_norm((w, h), b0))
                if (b1[0] > 0 and b1[3] < h-1) or (b1[0] == 0 and  b1[2]-b1[0] > px_thres) or (b1[3] == h-1 and b1[2]-b1[1] > px_thres):
                    b1_list.append(convert_norm((w, h), b1))
                if (b2[2] < w-1 and b2[1] > 0 ) or (b2[2] == w-1 and  b2[2]-b2[0] > px_thres) or (b2[1] == 0 and b2[3]-b2[1] > px_thres):
                    b2_list.append(convert_norm((w, h), b2))
                if (b3[0] > 0 and b3[1] > 0) or (b3[0] == 0 and  b3[2]-b3[0] > px_thres) or (b3[1] == 0 and b3[3]-b3[1] > px_thres):
                    b3_list.append(convert_norm((w, h), b3))


                # if b0[2]-b0[0] > px_thres and b0[3]-b0[1] > px_thres:
                #     b0_list.append(convert_norm((w, h), b0))
                # if b1[2]-b1[0] > px_thres and b1[3]-b1[1] > px_thres:
                #     b1_list.append(convert_norm((w, h), b1))
                # if b2[2]-b2[0] > px_thres and b2[3]-b2[1] > px_thres:
                #     b2_list.append(convert_norm((w, h), b2))
                # if b3[2]-b3[0] > px_thres and b3[3]-b3[1] > px_thres:
                #     b3_list.append(convert_norm((w, h), b3)) #cid, xc, yc, w, h, mid
            # if len(b0_list):
            f_txt = open(os.path.join(save_lbl_dir, name.split('.')[0] + '_i0j0.txt'), 'w')
            for i0 in b0_list:
                f_txt.write( "%s %s %s %s %s\n" % (0, i0[0], i0[1], i0[2], i0[3]))
            f_txt.close()
            # if len(b1_list):
            f_txt = open(os.path.join(save_lbl_dir, name.split('.')[0] + '_i0j1.txt'), 'w')
            for i1 in b1_list:
                # print('i1', i1)
                f_txt.write( "%s %s %s %s %s\n" % (0, i1[0], i1[1], i1[2], i1[3]))
            f_txt.close()
            # if len(b2_list):
            f_txt = open(os.path.join(save_lbl_dir, name.split('.')[0] + '_i1j0.txt'), 'w')
            for i2 in b2_list:
                f_txt.write( "%s %s %s %s %s\n" % (0, i2[0], i2[1], i2[2], i2[3]))
            f_txt.close()
            # if len(b3_list):
            f_txt = open(os.path.join(save_lbl_dir, name.split('.')[0] + '_i1j1.txt'), 'w')
            for i3 in b3_list: #cid, xc, yc, w, h, mid
                f_txt.write( "%s %s %s %s %s\n" % (0, i3[0], i3[1], i3[2], i3[3]))
            f_txt.close()


def split_syn_wnd_trn_val(comment='wnd', seed=17, pxwhr='px10whr6'):
    display_type = comment.split('_')[-1]
    syn_args.syn_data_dir = syn_args.syn_data_dir.format(display_type)
    step = syn_args.tile_size * syn_args.resolution
    all_files = np.sort(glob.glob(os.path.join(syn_args.syn_data_dir, '{}_all_images_step{}'.format(display_type, step), '*.png')))
    num_files = len(all_files)

    np.random.seed(seed)
    all_indices = np.random.permutation(num_files)
    data_txt_dir = syn_args.syn_txt_dir.format(display_type)
    if not os.path.exists(data_txt_dir):
        os.makedirs(data_txt_dir)
    print('data_txt_dir', data_txt_dir)

    trn_img_txt = open(os.path.join(data_txt_dir, '{}_train_img_seed{}.txt'.format(comment, seed)), 'w')
    trn_lbl_txt = open(os.path.join(data_txt_dir, '{}_train_lbl_seed{}.txt'.format(comment, seed)), 'w')

    val_img_txt = open(os.path.join(data_txt_dir, '{}_val_img_seed{}.txt'.format(comment, seed)), 'w')
    val_lbl_txt = open(os.path.join(data_txt_dir, '{}_val_lbl_seed{}.txt'.format(comment, seed)), 'w')

    num_val = int(num_files*syn_args.val_percent)
#    lbl_dir = os.path.join(syn_args.syn_annos_dir, 'minr{}_linkr{}_{}_{}_all_annos_txt_step{}'.format(syn_args.min_region, syn_args.link_r, pxwhr, display_type, step))
    lbl_dir = os.path.join(syn_args.syn_annos_dir, '{}_{}_all_annos_txt_step{}'.format(pxwhr, display_type, step))
    for i in all_indices[:num_val]:
        val_img_txt.write('%s\n' % all_files[i])
        val_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(all_files[i]).replace(IMG_FORMAT, TXT_FORMAT)))
    val_img_txt.close()
    val_lbl_txt.close()
    for j in all_indices[num_val:]:
        trn_img_txt.write('%s\n' % all_files[j])
        trn_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(all_files[j]).replace(IMG_FORMAT, TXT_FORMAT)))
    trn_img_txt.close()
    trn_lbl_txt.close()


def create_syn_data(comment='wnd', pxs='seed17'):
    dt = comment.split('_')[-1]
    data_txt_dir = syn_args.syn_txt_dir.format(dt)

    data_txt = open(os.path.join(data_txt_dir, '{}_{}.data'.format(comment, pxs)), 'w')
    data_txt.write('train={}/{}_train_img_{}.txt\n'.format(data_txt_dir, comment, pxs))
    data_txt.write('train_label={}/{}_train_lbl_{}.txt\n'.format(data_txt_dir, comment, pxs))

     #********** syn_0_xview_number corresponds to train*.py the number of train files
    df = pd.read_csv(os.path.join(data_txt_dir, '{}_train_img_{}.txt'.format(comment, pxs)), header=None)
    data_txt.write('syn_wnd_number={}\n'.format(df.shape[0]))
    data_txt.write('classes=%s\n' % str(syn_args.class_num))

    data_txt.write('valid={}/{}_val_img_{}.txt\n'.format(data_txt_dir, comment, pxs))
    data_txt.write('valid_label={}/{}_val_lbl_{}.txt\n'.format(data_txt_dir, comment, pxs))
    data_txt.write('names=./data_wnd/wnd.names\n')
    data_txt.write('backup=backup/\n')
    data_txt.write('eval={}'.format(comment))
    data_txt.close()


def sep_tif_bands(img_dir, save_dir):
    import rasterio
    from rasterio.plot import show

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file_list = glob.glob(os.path.join(img_dir, '*.tif'))
    for f in file_list:
        #to display RGB
        dataset = rasterio.open(f)
        # show(dataset.read([1,2,3]))
        name = os.path.basename(f)
        new_dataset = rasterio.open(os.path.join(save_dir, name), 'w', driver='GTiff',
                                    height=dataset.height, width=dataset.width, count=3,
                                    dtype=dataset.dtypes[0], transform=dataset.affine
                                    )
        new_dataset.write(dataset.read([1,2,3]))
        new_dataset.close()
        #to display just the red band:
        # show(dataset.read(1))


def get_args(cmt):
    parser = argparse.ArgumentParser()
    parser.add_argument("--syn_data_dir", type=str,
                        help="Path to folder containing raw synthetic images and annos ",
                        default='/data/users/yang/data/synthetic_data/{}'.format(cmt) + '_{}/')
                         # default='/media/lab/Yang/data/wind_turbine/')

    parser.add_argument("--syn_img_dir", type=str,
                        help="Path to folder containing cropped synthetic images ",
                        default='/data/users/yang/data/synthetic_data/{}/')
                        # default='/media/lab/Yang/data/wind_turbine_images/')
    parser.add_argument("--syn_annos_dir", type=str,
                        # default='/media/lab/Yang/data/wind_turbine_labels/',
                        default='/data/users/yang/data/synthetic_data/{}_txt_xcycwh/',
                        help="syn label.txt")
    parser.add_argument("--syn_box_dir", type=str,
                        # default='/media/lab/Yang/data/wind_turbine_bbox/',
                        default='/data/users/yang/data/synthetic_data/{}_gt_bbox/',
                        help="syn related txt files")

    parser.add_argument("--syn_txt_dir", type=str, default='/data/users/yang/code/yxu-yolov3-xview/data_wnd/{}'.format(cmt) + '_{}',
                        help="syn related txt files")
                        
    parser.add_argument("--real_txt_dir", type=str, default='/data/users/yang/code/yxu-yolov3-xview/data_wnd/',
                        help="syn related txt files")

    parser.add_argument("--syn_display_type", type=str, default='color',
                        help="texture, color, mixed")  # syn_color0, syn_texture0,
    #fixme ---***** min_region ***** change
    parser.add_argument("--min_region", type=int, default=100, help="300 100 the smallest #pixels (area) to form an object")
    parser.add_argument("--link_r", type=int, default=15,
                        help="the #pixels between two connected components to be grouped")
    parser.add_argument("--resolution", type=float, default=1, help="0.3 resolution of synthetic data")
    parser.add_argument("--tile_size", type=int, default=608, help="image size")
    parser.add_argument("--class_num", type=int, default=1, help="class number")
    parser.add_argument("--val_percent", type=float, default=0.25, help="train:val=0.75:0.25")


    args = parser.parse_args()
    args.syn_annos_dir = args.syn_annos_dir.format(cmt)
#    args.syn_txt_dir = args.syn_txt_dir.format(cmt)
    args.syn_box_dir = args.syn_box_dir.format(cmt)

    if not os.path.exists(args.syn_annos_dir):
        os.makedirs(args.syn_annos_dir)
#    if not os.path.exists(args.syn_txt_dir):
#        os.makedirs(args.syn_txt_dir)
    if not os.path.exists(args.syn_box_dir):
        os.makedirs(args.syn_box_dir)


    return args




if __name__ == "__main__":
    syn_args = get_args()

    '''
    process bkg tifs
    '''
    # img_dir = '/media/lab/Yang/data/uspp_naip/'
    # save_dir = '/media/lab/Yang/data/uspp_naip_rgb/'
    # sep_tif_bands(img_dir, save_dir)

    '''
    crop windturbine images
    recreate corresponging labels
    '''
    # resize_crop_windturbine(px_thres=10)

    '''
    check new lables
    plot bbox on images
    '''
    # lbl_file =  os.path.join(syn_args.syn_annos_dir, 'naip_915_CA_WND_i0j1.txt')
    # # lbl_file = os.path.join(syn_args.syn_annos_dir, 'naip_1101_CA_WND_i0j0.txt')
    # img_dir = syn_args.syn_img_dir
    #
    # # lbl_file = os.path.join(syn_args.syn_data_dir, 'naip_915_CA_WND.txt')
    # # # lbl_file = os.path.join(syn_args.syn_data_dir, 'naip_324_AZ_WND.txt')
    # # img_dir = syn_args.syn_data_dir
    #
    # save_dir = syn_args.syn_box_dir
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # name = os.path.basename(lbl_file)
    # gbc.plot_img_with_bbx(os.path.join(img_dir, name.replace('.txt', '.jpg')), lbl_file, save_dir)


#    IMGTPYE = '.jpg'
#    IMGTPYE = '.png'
#    cmt = 'syn_uspp_bkg_shdw_split_scatter_gauss_rotate_wnd_v2'
#    syn_args = get_args(cmt)
    '''
    split train val
    '''
    # split_syn_wnd_trn_val(comment='wnd', seed=17, pxs='px10_seed17')
#    comment = 'syn_uspp_bkg_shdw_split_scatter_gauss_rotate_wnd_v2_color'
#    split_syn_wnd_trn_val(comment, seed=17, pxwhr='px10whr6')

    '''
    create *.data
    '''
#    comment = 'wnd'
#    comment = 'syn_uspp_bkg_shdw_split_scatter_gauss_rotate_wnd_v2_color'
#    create_syn_data(comment, pxs='seed17')








