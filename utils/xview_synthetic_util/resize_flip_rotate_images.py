import os
from PIL import Image
import shutil
import argparse
import math
import glob
import pandas as pd
from utils.xview_synthetic_util import process_syn_xview_background_wv_split as psx
from utils.object_score_util import get_bbox_coords_from_annos_with_object_score as gbc



def flip_images(img_name, src_dir=None):
    img = Image.open(os.path.join(src_dir, img_name))
    name_str = img_name.split('.')[0]
    save_dir = src_dir + '_flip'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    shutil.copy(os.path.join(src_dir, img_name), os.path.join(save_dir, img_name))

    out_lr_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    out_lr_flip.save(os.path.join(save_dir,'{}_lr.jpg'.format(name_str)))

def rotate_images(img_name, src_dir=None):
    img = Image.open(os.path.join(src_dir, img_name))
    name_str = img_name.split('.')[0]
    save_dir = src_dir + '_aug'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    shutil.copy(os.path.join(src_dir, img_name), os.path.join(save_dir, img_name))

    out_rt_90 = img.transpose(Image.ROTATE_90)
    out_rt_90.save(os.path.join(save_dir,'{}_rt90.jpg'.format(name_str)))
    out_rt_180 = img.transpose(Image.ROTATE_180)
    out_rt_180.save(os.path.join(save_dir,'{}_rt180.jpg'.format(name_str)))
    out_rt_270 = img.transpose(Image.ROTATE_270)
    out_rt_270.save(os.path.join(save_dir,'{}_rt270.jpg'.format(name_str)))



def flip_rotate_images(img_name, src_dir=None):
    if src_dir:
        img = Image.open(os.path.join(src_dir, img_name))
        name_str = img_name.split('.')[0]
        shutil.copy(os.path.join(src_dir, img_name), os.path.join(save_dir, img_name))

        out_lr_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        out_lr_flip.save(os.path.join(save_dir,'{}_lr.jpg'.format(name_str)))
        # out_tb_flip = img.transpose(Image.FLIP_TOP_BOTTOM)
        # out_tb_flip.save(os.path.join(save_dir,'{}_tb.jpg'.format(name_str)))
        out_rt_90 = img.transpose(Image.ROTATE_90)
        out_rt_90.save(os.path.join(save_dir,'{}_rt90.jpg'.format(name_str)))
        out_rt_180 = img.transpose(Image.ROTATE_180)
        out_rt_180.save(os.path.join(save_dir,'{}_rt180.jpg'.format(name_str)))
        out_rt_270 = img.transpose(Image.ROTATE_270)
        out_rt_270.save(os.path.join(save_dir,'{}_rt270.jpg'.format(name_str)))
    else:
        img_name = img_name.split('.')[0]
        img = Image.open(args.images_save_dir[:-1] + '_of_{}/{}.jpg'.format(img_name, img_name))
        out_lr_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        out_lr_flip.save(args.images_save_dir[:-1] + '_of_{}/{}_lr.jpg'.format(img_name, img_name))
        # out_tb_flip = img.transpose(Image.FLIP_TOP_BOTTOM)
        # out_tb_flip.save(args.images_save_dir[:-1] + '_of_{}/{}_tb.jpg'.format(img_name, img_name))
        out_rt_90 = img.transpose(Image.ROTATE_90)
        out_rt_90.save(args.images_save_dir[:-1] + '_of_{}/{}_rt90.jpg'.format(img_name, img_name))
        out_rt_180 = img.transpose(Image.ROTATE_180)
        out_rt_180.save(args.images_save_dir[:-1] + '_of_{}/{}_rt180.jpg'.format(img_name, img_name))
        out_rt_270 = img.transpose(Image.ROTATE_270)
        out_rt_270.save(args.images_save_dir[:-1] + '_of_{}/{}_rt270.jpg'.format(img_name, img_name))



def get_rotated_point(x,y,angle):
    '''
    https://blog.csdn.net/weixin_44135282/article/details/89003793
    https://blog.csdn.net/guyuealian/article/details/78288131
    '''
    # (h, w) = image.shape[:2]
    # # 将图像中心设为旋转中心
    w, h = 1, 1
    (cX, cY) = (0.5, 0.5)

    #假设图像的宽度x高度为col*row, 图像中某个像素P(x1, y1)，绕某个像素点Q(x2, y2)
    #旋转θ角度后, 则该像素点的新坐标位置为(x, y)，其计算公式为：

    x = x
    y = h - y
    cX = cX
    cY = h - cY
    new_x = (x - cX) * math.cos(math.pi / 180.0 * angle) - (y - cY) * math.sin(math.pi / 180.0 * angle) + cX
    new_y = (x - cX) * math.sin(math.pi / 180.0 * angle) + (y - cY) * math.cos(math.pi / 180.0 * angle) + cY
    new_x = new_x
    new_y = h - new_y
    # return round(new_x), round(new_y) #四舍五入取整
    return new_x, new_y

def get_flipped_point(x, y, flip='tb'):
    w, h = 1, 1
    if flip == 'tb':
        new_y = h - y
        new_x = x
    elif flip == 'lr':
        new_x = w - x
        new_y = y
    return new_x, new_y


def flip_rotate_coordinates(img_dir, lbl_dir, save_dir, aug_dir, name, angle=0, flip=''):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # backup the original lbl first
    shutil.copy(os.path.join(lbl_dir, '{}.txt'.format(name)), os.path.join(aug_dir, '{}.txt'.format(name)))
    if angle:
        lbl_file = os.path.join(aug_dir, '{}_rt{}.txt'.format(name, angle))
    elif flip:
        lbl_file = os.path.join(aug_dir, '{}_{}.txt'.format(name, flip))
    shutil.copy(os.path.join(lbl_dir, '{}.txt'.format(name)), lbl_file)
    df_lf = pd.read_csv(lbl_file, header=None, sep=' ')
    for i in range(df_lf.shape[0]):
        if angle:
            df_lf.loc[i, 1], df_lf.loc[i, 2] = get_rotated_point(df_lf.loc[i, 1], df_lf.loc[i, 2], angle)
            if angle==90 or angle == 270:
                w = df_lf.loc[i, 3]
                h = df_lf.loc[i, 4]
                df_lf.loc[i, 3], df_lf.loc[i, 4] = h, w
        elif flip:
            df_lf.loc[i, 1], df_lf.loc[i, 2] = get_flipped_point(df_lf.loc[i, 1], df_lf.loc[i, 2], flip)
    df_lf.to_csv(lbl_file, header=False, index=False, sep=' ')
    name = os.path.basename(lbl_file)
    print('name', name)
    img_name = name.replace('.txt', '.jpg')
    img_file = os.path.join(img_dir, img_name)
    # print('img_file', img_file)
    gbc.plot_img_with_bbx(img_file, lbl_file, save_path=save_dir)


def create_data_for_augment_img_lables(img_names, eh_type):
    shutil.copy(os.path.join(args.data_save_dir, 'xviewtest_img_px23whr3_seed17_m4_rc1_{}.txt'.format(eh_type)),
                os.path.join(args.data_save_dir, 'xviewtest_img_px23whr3_seed17_m4_rc1_{}_aug.txt'.format(eh_type)))
    shutil.copy(os.path.join(args.data_save_dir, 'xviewtest_lbl_px23whr3_seed17_m4_rc1_{}.txt'.format(eh_type)),
                os.path.join(args.data_save_dir, 'xviewtest_lbl_px23whr3_seed17_m4_rc1_{}_aug.txt'.format(eh_type)))
    val_img_file = open(os.path.join(args.data_save_dir, 'xviewtest_img_px23whr3_seed17_m4_rc1_{}_aug.txt'.format(eh_type)), 'a')
    val_lbl_file = open(os.path.join(args.data_save_dir, 'xviewtest_lbl_px23whr3_seed17_m4_rc1_{}_aug.txt'.format(eh_type)), 'a')
    for img_name in img_names:
        img_dir = args.images_save_dir[:-1] + '_of_{}/'.format(img_name)
        lbl_dir = args.annos_save_dir[:-1] + '_val_m4_rc1_{}/'.format(img_name)
        img_files = glob.glob(os.path.join(img_dir, '{}_*.jpg'.format(img_name)))
        for f in img_files:
            name = os.path.basename(f)
            val_img_file.write('%s\n' % f)
            val_lbl_file.write('%s\n' % os.path.join(lbl_dir, name.replace('.jpg', '.txt')))

    psx.create_xview_base_data_for_onemodel_aug_easy_hard(model_id=4, rc_id=1, eh_type=eh_type, base_cmt='px23whr3_seed17')



def get_args(px_thres=None, whr_thres=None, seed=17):
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_save_dir", type=str, help="to save chip trn val images files",
                        default='/media/lab/Yang/data/xView_YOLO/images/')

    parser.add_argument("--txt_save_dir", type=str, help="to save  related label files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')

    parser.add_argument("--data_save_dir", type=str, help="to save data files",
                        default='/media/lab/Yang/code/yolov3/data_xview/{}_cls/')
    parser.add_argument("--annos_save_dir", type=str, help="to save txt annotation files",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')
    parser.add_argument("--cat_sample_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/')
    parser.add_argument("--font3", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 10}")
    parser.add_argument("-ft2", "--font2", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 13}")
    parser.add_argument("--class_num", type=int, default=1, help="Number of Total Categories")  # 60  6 1
    parser.add_argument("--input_size", type=int, default=608, help="Number of Total Categories")  # 300 416

    args = parser.parse_args()
    #fixme ----------==--------change
    args.images_save_dir = args.images_save_dir + '{}_{}cls/'.format(args.input_size, args.class_num)
    if px_thres:
        args.data_save_dir = args.data_save_dir.format(args.class_num) + 'px{}whr{}_seed{}/'.format(px_thres, whr_thres, seed)
        args.annos_save_dir = args.annos_save_dir + '{}/{}_cls_xcycwh_px{}whr{}/'.format(args.input_size, args.class_num, px_thres, whr_thres)
    else:
        args.data_save_dir = args.data_save_dir.format(args.class_num)
        args.annos_save_dir = args.annos_save_dir + '{}/{}_cls_xcycwh/'.format(args.input_size, args.class_num)
    args.txt_save_dir = args.txt_save_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)
    args.cat_sample_dir = args.cat_sample_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)
    if not os.path.exists(args.txt_save_dir):
        os.makedirs(args.txt_save_dir)

    if not os.path.exists(args.annos_save_dir):
        os.makedirs(args.annos_save_dir)

    if not os.path.exists(args.images_save_dir):
        os.makedirs(args.images_save_dir)
    if not os.path.exists(args.cat_sample_dir):
        os.makedirs(args.cat_sample_dir)
    return args


if __name__ == '__main__':
    whr_thres = 3
    px_thres = 23
    seed = 17
    args = get_args(px_thres, whr_thres, seed)

    '''
    flip and rotate images 
    left-right flip, tob-bottom flip
    rotate90, rotate 180, rotate 270
    '''
    # # img_name = '2315_359'
    # img_name = '2315_329'
    # flip_rotate_images(img_name)

    # typ = 'val'
    typ = 'trn'
    img_dir = args.images_save_dir[:-1] + '_rc_{}_new_ori_multi'.format(typ)
    img_list = glob.glob(os.path.join(img_dir, '*.jpg'))
    print('img_list', len(img_list))
    for f in img_list:
        name = os.path.basename(f)
        img_dir = os.path.dirname(f)
        flip_rotate_images(img_name=name, src_dir=img_dir)

    '''
    flip and rotate coordinates of bbox 
    left-right flip, tob-bottom flip
    rotate90, rotate 180, rotate 270
    ** manually create '/media/lab/Yang/data/xView_YOLO/labels/608/1_cls_xcycwh_px23whr3_val_m4_rc1_{}/'.format(img_name)
    '''
    # lbl_dir = args.annos_save_dir[:-1] + '_val_m4_rc1_{}/'.format(name)
    # img_dir = args.images_save_dir[:-1] + '_of_{}/'.format(name)
    # img_name = '2315_359'
    # img_name = '2315_329'
    #
    # angle = 270
    # # angle = 180
    # # angle = 90
    # flip_rotate_coordinates(img_dir, lbl_dir, img_name, angle=angle)

    # flip = 'tb'
    # flip = 'lr'
    # flip_rotate_coordinates(img_dir, lbl_dir, img_name, flip=flip)

    # typ = 'val'
    typ = 'trn'
    img_dir = args.images_save_dir[:-1] + '_rc_{}_new_ori_multi_aug'.format(typ)
    lbl_dir = args.annos_save_dir[:-1] + '_rc_{}_new_ori_multi_rcid'.format(typ)
    save_dir = args.cat_sample_dir + 'image_with_bbox/{}_aug/'.format(name)
    lbl_list = glob.glob(os.path.join(lbl_dir, '*.txt'))
    for lbl in lbl_list:
        name = os.path.basename(lbl).split('.')[0]
        angle_list = [90, 180, 270]
        for angle in angle_list:
            flip_rotate_coordinates(img_dir, lbl_dir, save_dir, name, angle=angle)
        flip_list = ['lr'] # 'tb
        for flip in flip_list:
            flip_rotate_coordinates(img_dir, lbl_dir, save_dir, name, flip=flip)


    '''
    add augmented images and labels into val file
    create corresponding *.data
    '''
    # # img_name = '2315_359'
    # img_names = ['2315_329', '2315_359']
    # # eh_type = 'hard'
    # eh_type = 'easy'
    # create_data_for_augment_img_lables(img_names, eh_type)

