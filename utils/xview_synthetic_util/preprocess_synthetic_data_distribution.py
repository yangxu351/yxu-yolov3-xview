import glob
import numpy as np
import argparse
import os
from skimage import io, color
import utils.wv_util as wv
from utils.utils_xview import coord_iou, compute_iou
from utils_object_score import get_bbox_coords_from_annos_with_object_score as gbc
from utils.data_process_distribution_vis_util import process_wv_coco_for_yolo_patches_no_trnval as pwv
import pandas as pd
from ast import literal_eval
from matplotlib import pyplot as plt
import json
import shutil

IMG_FORMAT = 'jpg'
TXT_FORMAT = 'txt'


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


def analyze_category_distribution_by_catid(catid, txt_path, json_number_file, json_name_file, px_thresh=4, whr_thres=3):
    txt_files = np.sort(glob.glob(os.path.join(txt_path, "*.{}".format(TXT_FORMAT))))
    plane_number_2_img_number_map = {}
    plane_number_2_txt_name_map = {}
    for i in range(len(txt_files)):
        if not is_non_zero_file(txt_files[i]):
            continue
        df_txt = pd.read_csv(txt_files[i], header=None, delimiter=' ').to_numpy()
        if catid not in np.unique(df_txt[:, 0]):
            continue
        df_txt = df_txt[df_txt[:, 0] == catid]
        df_txt[:, 1:] = df_txt[:, 1:] * syn_args.tile_size
        df_txt = df_txt[df_txt[:, 3] > px_thresh]
        df_txt = df_txt[df_txt[:, 4] > px_thresh]
        df_wh = np.vstack((df_txt[:, 3] / df_txt[:, 4], df_txt[:, 4] / df_txt[:, 3]))
        df_txt = df_txt[np.max(df_wh, axis=0) <= whr_thres]
        # print(df_txt.shape)
        if df_txt.shape[0] == 0:
            continue
        plane_number = np.count_nonzero(df_txt[:, 0] == catid)
        if plane_number not in plane_number_2_img_number_map.keys():
            plane_number_2_img_number_map[plane_number] = 1
            plane_number_2_txt_name_map[plane_number] = [txt_files[i].split('/')[-1]]
        else:
            plane_number_2_img_number_map[plane_number] += 1
            plane_number_2_txt_name_map[plane_number].append(txt_files[i].split('/')[-1])
    json.dump(plane_number_2_img_number_map, open(json_number_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)
    json.dump(plane_number_2_txt_name_map, open(json_name_file, 'w'), ensure_ascii=False, indent=2, cls=MyEncoder)


def draw_bar_of_image_numbers_for_certain_number_of_planes(png_name, cat_distribution_map, legend='xView'):
    '''
    x-axis: the number of planes
    y-axis: the number of images
    :param catid:
    :param cat_distribution_map:
    :return:
    '''
    if legend == 'xView':
        args = pwv.get_args()
    else:
        args = get_syn_args()
    num_planes = np.array([int(k) for k in cat_distribution_map.keys()])
    num_planes_sort_indices = np.argsort(num_planes)
    num_images = np.array([v for v in cat_distribution_map.values()])

    save_dir = args.txt_save_dir + 'data_distribution_fig/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    title = "Airplane Distribution (chips)"
    labels = num_planes[num_planes_sort_indices]

    x = num_planes[num_planes_sort_indices]
    ylist = num_images[num_planes_sort_indices]

    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    fig, ax = plt.subplots(1, 1)
    width = 0.35
    rects = ax.bar(np.array(x) - width / 2, ylist, width, label=legend)  # , label=labels
    autolabel(ax, rects, x, labels, ylist, rotation=0)
    ax.legend()
    xlabel = 'Number of Airplanes'
    ylabel = "Number of Images"
    plt.title(title, literal_eval(args.font2))
    plt.ylabel(ylabel, literal_eval(args.font2))
    plt.xlabel(xlabel, literal_eval(args.font2))
    plt.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.grid()
    plt.savefig(os.path.join(save_dir, png_name))
    plt.show()


def compare_bar_of_image_numbers_for_certain_number_of_planes(png_name, cat_distribution_map, cat_distribution_map_syn):
    '''
    x-axis: the number of planes
    y-axis: the number of images
    :param catid:
    :param cat_distribution_map:
    :return:
    '''
    num_planes = np.array([int(k) for k in cat_distribution_map.keys()])
    num_planes_sort_indices = np.argsort(num_planes)
    num_planes = num_planes[num_planes_sort_indices]
    num_images = np.array([v for v in cat_distribution_map.values()])
    num_images = num_images[num_planes_sort_indices]

    num_planes_syn = np.array([int(k) for k in cat_distribution_map_syn.keys()])
    num_planes_syn_sort_indices = np.argsort(num_planes_syn)
    num_planes_syn = num_planes_syn[num_planes_syn_sort_indices]
    num_images_syn = np.array([v for v in cat_distribution_map_syn.values()])
    num_images_syn = num_images_syn[num_planes_syn_sort_indices]

    save_dir = args.txt_save_dir + 'data_distribution_fig/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    title = "Comparison Airplane Distribution (chips)"

    labels = num_planes.tolist()
    x = num_planes
    ylist = num_images

    labels_syn = num_planes_syn.tolist()
    x_syn = num_planes_syn
    ylist_syn = num_images_syn

    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    fig, ax = plt.subplots(1, 1)
    width = 0.35

    rects_syn = ax.bar(np.array(x_syn) + width / 2, ylist_syn, width, label='Synthetic')  # , label=labels
    autolabel(ax, rects_syn, x_syn, labels_syn, ylist_syn, rotation=0)

    rects = ax.bar(np.array(x) - width / 2, ylist, width, label='xView')  # , label=labels
    autolabel(ax, rects, x, labels, ylist, rotation=0)
    ax.legend()

    xlabel = 'Number of Airplanes'
    ylabel = "Number of Images"
    plt.title(title, literal_eval(args.font2))
    plt.ylabel(ylabel, literal_eval(args.font2))
    plt.xlabel(xlabel, literal_eval(args.font2))
    plt.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.grid()
    plt.savefig(os.path.join(save_dir, png_name))
    plt.show()


def clean_image_files(file_path, cities, streets, tile_size=608, resolution=0.3, white_thresh=0.5):
    '''
    remove rgb images those contain more than white_thresh*100% white pixels
    :param file_path:
    :param cities:
    :param streets:
    :param tile_size:
    :param resolution:
    :param white_thresh:
    :return:
    '''
    step = tile_size * resolution
    image_folder_name = '{}_{}_{}_images_step{}'
    label_folder_name = '{}_{}_{}_annos_step{}'

    for i in range(len(cities)):
        image_path = os.path.join(file_path,
                                  image_folder_name.format(args.syn_display_type, cities[i], streets[i], step))
        image_files = np.sort(glob.glob(os.path.join(image_path, '*.{}'.format(IMG_FORMAT))))
        image_names = [os.path.basename(i) for i in image_files]
        label_path = os.path.join(file_path,
                                  label_folder_name.format(args.syn_display_type, cities[i], streets[i], step))
        for ix, f in enumerate(image_files):
            img = io.imread(f)
            img = color.rgb2gray(img)
            white_num = np.sum(img == 1)
            white_ratio = white_num / img.shape[0] / img.shape[1]

            label_file = os.path.join(label_path, image_names[ix])
            gt = io.imread(label_file)
            gt = color.rgb2gray(gt)
            gt_white_num = np.sum(gt == 1)
            gt_white_ratio = gt_white_num / gt.shape[0] / gt.shape[1]
            if white_ratio > white_thresh or gt_white_ratio == 1:
                os.remove(image_files[ix])
                os.remove(label_file)


def rename_images_groundtruth(cities, streets, tile_size=608, resolution=0.3):
    rgb_suffix = '_RGB'
    gt_suffix = '_GT'
    step_size = tile_size * resolution
    for i in range(len(cities)):
        rgb_path = os.path.join(args.syn_plane_img_anno_dir,
                                'syn_{}_{}_images_step{}'.format(cities[i], streets[i], step_size))
        gt_path = os.path.join(args.syn_plane_img_anno_dir,
                               'syn_{}_{}_annos_step{}'.format(cities[i], streets[i], step_size))
        rgb_files = np.sort(glob.glob(os.path.join(rgb_path, '*.{}'.format(IMG_FORMAT))))
        rgb_names = [os.path.basename(f) for f in rgb_files]

        gt_files = np.sort(glob.glob(os.path.join(gt_path, '*.{}'.format(IMG_FORMAT))))
        gt_names = [os.path.basename(f) for f in gt_files]

        for ix, f in enumerate(rgb_files):
            new_name = rgb_names[ix].replace(rgb_suffix, '')
            os.rename(f, os.path.join(rgb_path, new_name))

        for ix, f in enumerate(gt_files):
            new_name = gt_names[ix].replace(gt_suffix, '')
            os.rename(f, os.path.join(gt_path, new_name))


def merge_txt_or_rgb_files(merge_type='txt', copy=True):
    step = args.tile_size * args.resolution
    if merge_type == 'txt':
        dst_dir = os.path.join(args.syn_plane_txt_dir,
                               'minr{}_linkr{}_{}_all_annos_step{}'.format(args.min_region, args.link_r,
                                                                           args.syn_display_type, step))
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        for i in range(len(args.cities)):
            folder_name = 'minr{}_linkr{}_{}_{}_{}_annos_step{}'.format(args.min_region, args.link_r,
                                                                        args.syn_display_type, args.cities[i],
                                                                        args.streets[i], step)
            src_files = glob.glob(os.path.join(args.syn_plane_txt_dir, folder_name, '*.{}'.format(TXT_FORMAT)))

            for sf in src_files:
                shutil.copy(sf, dst_dir)
                if copy:
                    shutil.copy(sf, args.syn_annos_save_dir)
    else:
        dst_dir = os.path.join(args.syn_plane_img_anno_dir, '{}_all_images_step{}'.format(args.syn_display_type, step))
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        for i in range(len(args.cities)):
            folder_name = '{}_{}_{}_images_step{}'.format(args.syn_display_type, args.cities[i], args.streets[i], step)
            src_files = glob.glob(os.path.join(args.syn_plane_img_anno_dir, folder_name, '*.{}'.format(IMG_FORMAT)))
            for sf in src_files:
                shutil.copy(sf, dst_dir)
                if copy:
                    shutil.copy(sf, args.syn_images_save_dir)


def get_syn_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_save_dir", type=str, help="to save chip trn val images files",
                        default='/media/lab/Yang/data/xView_YOLO/images/')

    parser.add_argument("--syn_plane_img_anno_dir", type=str, help="images and annotations of synthetic airplanes",
                        default='/media/lab/Yang/data/synthetic_data/Airplanes/')

    parser.add_argument("--syn_display_type", type=str, help="syn_texture, syn_color, syn_mixed",
                        default='syn_texture')

    parser.add_argument("--syn_plane_txt_dir", type=str, help="txt labels of synthetic airplanes",
                        default='/media/lab/Yang/data/synthetic_data/Airplanes_txt_xcycwh/')

    parser.add_argument("--syn_plane_gt_bbox_dir", type=str, help="gt images with bbox of synthetic airplanes",
                        default='/media/lab/Yang/data/synthetic_data/Airplanes_gt_bbox/')

    parser.add_argument("--syn_images_save_dir", type=str, help="rgb images of synthetic airplanes",
                        default='/media/lab/Yang/data/xView_YOLO/images/')
    parser.add_argument("--syn_annos_save_dir", type=str, help="gt of synthetic airplanes",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')
    parser.add_argument("--syn_txt_save_dir", type=str, help="gt related files of synthetic airplanes",
                        default='/media/lab/Yang/data/xView_YOLO/labels/')

    parser.add_argument("--data_save_dir", type=str, help="to save data files",
                        default='../../data_xview/{}_cls/')

    parser.add_argument("--results_dir", type=str, help="to save category files",
                        default='../../result_output/{}_cls/')

    parser.add_argument("--cat_sample_dir", type=str, help="to save figures",
                        default='/media/lab/Yang/data/xView_YOLO/cat_samples/')

    parser.add_argument("--font3", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 10}")
    parser.add_argument("-ft2", "--font2", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 13}")
    parser.add_argument("--class_num", type=int, default=1, help="Number of Total Categories")  # 60  6
    parser.add_argument("--one_syn", type=str, default='1syn', help="one syn Number of Total Categories")
    parser.add_argument("--seed", type=int, default=1024, help="random seed")
    parser.add_argument("--tile_size", type=int, default=608, help="image size")  # 300 416

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
    syn_args.syn_images_save_dir += '{}_{}/'.format(syn_args.tile_size, syn_args.one_syn)
    syn_args.syn_annos_save_dir += '{}/{}_cls_xcycwh/'.format(syn_args.tile_size, syn_args.one_syn)
    syn_args.syn_txt_save_dir += '{}/{}_cls/'.format(syn_args.tile_size, syn_args.one_syn)
    if not os.path.exists(syn_args.syn_images_save_dir):
        os.makedirs(syn_args.syn_images_save_dir)

    if not os.path.exists(syn_args.syn_annos_save_dir):
        os.makedirs(syn_args.syn_annos_save_dir)

    if not os.path.exists(syn_args.syn_txt_save_dir):
        os.makedirs(syn_args.syn_txt_save_dir)

    syn_args.syn_plane_img_anno_dir = syn_args.syn_plane_img_anno_dir + syn_args.syn_display_type + '/'
    syn_args.syn_plane_txt_dir = syn_args.syn_plane_txt_dir + syn_args.syn_display_type + '/'
    syn_args.syn_plane_gt_bbox_dir = syn_args.syn_plane_gt_bbox_dir + syn_args.syn_display_type + '/'
    if not os.path.exists(syn_args.syn_plane_img_anno_dir):
        os.mkdir(syn_args.syn_plane_img_anno_dir)

    if not os.path.exists(syn_args.syn_plane_txt_dir):
        os.mkdir(syn_args.syn_plane_txt_dir)

    if not os.path.exists(syn_args.syn_plane_gt_bbox_dir):
        os.mkdir(syn_args.syn_plane_gt_bbox_dir)
    return syn_args


if __name__ == "__main__":
    syn_args = get_syn_args()

    '''
    count aircraft distribution: count the number of images that contain certain number of planes
    '''
    # args = pwv.get_args()
    # catid = 0
    # whr_thres = 4 # 3
    # px_thresh = 6 # 4
    # txt_path = args.annos_save_dir
    # # txt_path = args.syn_plane_txt_dir
    # json_number_file = os.path.join(args.txt_save_dir,
    #                          'xView_number_of_cat_{}_to_imagenumber_map_inputsize{}.json'.format(catid, args.tile_size))
    # json_txt_file = os.path.join(args.txt_save_dir,
    #                          'xView_number_of_cat_{}_to_imagetxt_map_inputsize{}.json'.format(catid, args.tile_size))
    # analyze_category_distribution_by_catid(catid, txt_path, json_number_file, json_txt_file, px_thresh, whr_thres)

    '''
    plot count aircraft distribution
    '''
    # args = pwv.get_args()
    # catid = 0
    # png_name = 'xView_cat_{}_nubmers_imagesnumber_dis.png'.format(catid)
    # cat_distribution_map = json.load(open(os.path.join(args.txt_save_dir,
    #                                                    'xView_number_of_cat_{}_to_imagenumber_map_inputsize{}.json'.format(
    #                                                        catid, args.tile_size))))
    # draw_bar_of_image_numbers_for_certain_number_of_planes(png_name, cat_distribution_map, 'xView')

    '''
    rename all the images and annos with suffix '_RGB' '_GT' 
    '''
    # rename_images_groundtruth(syn_args.cities, syn_args.streets, syn_args.tile_size, syn_args.resolution)

    '''
    remove labels that contain more than 80% white pixels
    clean up useless images
    '''
    # white_thresh = 0.5
    # clean_image_files(syn_args.syn_plane_img_anno_dir, syn_args.cities, syn_args.streets, syn_args.tile_size, syn_args.resolution, white_thresh)

    '''
    group annotation files, generate bbox for each object, and draw bbox for each ground truth files
    '''
    # step = syn_args.tile_size*syn_args.resolution
    # for i in range(len(syn_args.cities)):
    #     folder_name = '{}_'.format(syn_args.syn_display_type) + syn_args.cities[i] + '_{}'.format(syn_args.streets[i]) +'_annos_step{}'.format(step)
    #     lbl_path = os.path.join(syn_args.syn_plane_img_anno_dir, folder_name)
    #     save_txt_path = os.path.join(syn_args.syn_plane_txt_dir,  'minr{}_linkr{}_'.format(syn_args.min_region, syn_args.link_r) + folder_name)
    #     if not os.path.exists(save_txt_path):
    #         os.makedirs(save_txt_path)
    #     gbc.get_object_bbox_after_group(lbl_path, save_txt_path, class_label=0, min_region=syn_args.min_region, link_r=syn_args.link_r)
    #
    #     gt_files = np.sort(glob.glob(os.path.join(lbl_path, '*.{}'.format(IMG_FORMAT))))
    #     save_bbx_path = os.path.join(syn_args.syn_plane_gt_bbox_dir,  'minr{}_linkr{}_'.format(syn_args.min_region, syn_args.link_r) + folder_name)
    #     if not os.path.exists(save_bbx_path):
    #         os.makedirs(save_bbx_path)
    #     for g in gt_files:
    #         gt_name = g.split('/')[-1]
    #         txt_name = gt_name.replace(IMG_FORMAT, TXT_FORMAT)
    #         txt_file = os.path.join(save_txt_path, txt_name)
    #         gbc.plot_img_with_bbx(g, txt_file, save_bbx_path)

    '''
    merge all txt files or rgb images into one folder and copy to ...
    '''
    # merge_type='txt'
    # # merge_type='rgb'
    # copy = True
    # merge_txt_or_rgb_files(merge_type, copy)

    '''
    analyze synthetic data distribution
    '''
    # catid = 0
    # whr_thres = 4  # 3
    # px_thresh = 6  # 4
    # # txt_path = syn_args.annos_save_dir
    # step = syn_args.tile_size * syn_args.resolution
    # json_number_file = os.path.join(syn_args.syn_plane_txt_dir,
    #                                 '{}_number_of_cat_{}_to_imagenumber_map_inputsize{}.json'.format(
    #                                     syn_args.syn_display_type, catid, syn_args.tile_size))
    # json_name_file = os.path.join(syn_args.syn_plane_txt_dir,
    #                               '{}_number_of_cat_{}_to_imagetxt_map_inputsize{}.json'.format(syn_args.syn_display_type,
    #                                                                                             catid, syn_args.tile_size))
    # for i in range(len(syn_args.cities)):
    #     txt_path = os.path.join(syn_args.syn_plane_txt_dir,
    #                             'minr{}_linkr{}_{}_all_annos_step{}'.format(syn_args.min_region, syn_args.link_r,
    #                                                                         syn_args.syn_display_type, step))
    #
    #     analyze_category_distribution_by_catid(catid, txt_path, json_number_file, json_name_file, px_thresh, whr_thres)

    '''
    plot synthetic data aircraft distribution
    '''
    # catid = 0
    # png_name = '{}_cat_{}_nubmers_imagesnumber_dis.png'.format(syn_args.syn_display_type, catid)
    # cat_distribution_map = json.load(open(os.path.join(syn_args.syn_plane_txt_dir,
    #                          '{}_number_of_cat_{}_to_imagenumber_map_inputsize{}.json'.format(syn_args.syn_display_type, catid, syn_args.tile_size))))
    #
    # draw_bar_of_image_numbers_for_certain_number_of_planes(png_name, cat_distribution_map, legend='{}'.format(syn_args.syn_display_type))

    ''''
    compare xview and synthetic aircraft distribution
    '''
    # args = pwv.get_args()
    # catid = 0
    # png_name = 'xview_vs_{}_cat_{}_nubmers_imagesnumber_dis.png'.format(syn_args.syn_display_type, catid)
    # cat_distribution_map = json.load(open(os.path.join(args.txt_save_dir,
    #                                                    'xView_number_of_cat_{}_to_imagenumber_map_inputsize{}.json'.format(
    #                                                        catid, args.tile_size))))
    # cat_distribution_map_syn = json.load(open(os.path.join(syn_args.syn_plane_txt_dir,
    #                                                        '{}_number_of_cat_{}_to_imagenumber_map_inputsize{}.json'.format(
    #                                                            syn_args.syn_display_type, catid, syn_args.tile_size))))
    # compare_bar_of_image_numbers_for_certain_number_of_planes(png_name, cat_distribution_map, cat_distribution_map_syn)
