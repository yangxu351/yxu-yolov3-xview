import numpy as np
from matplotlib import pyplot as plt
from utils.xview_synthetic_util import preprocess_synthetic_data_distribution as pps
import os
from glob import glob
from skimage.color import rgb2gray
from skimage import io
import shutil




def compare_overlap():
    # file1 = '/media/lab/Yang/data/synthetic_data/Airplanes/syn_radial_200_images_test_ts608/airplanes_radial_200_59_RGB.jpg'
    # file2 = '/media/lab/Yang/data/synthetic_data/Airplanes/syn_radial_200_images_test_ts608/airplanes_radial_200_60_RGB.jpg'
    file1 = '/media/lab/Yang/data/synthetic_data/Airplanes/syn_radial_200_images_test_ts608/airplanes_radial_200_75_RGB.jpg'
    file2 = '/media/lab/Yang/data/synthetic_data/Airplanes/syn_radial_200_images_test_ts608/airplanes_radial_200_76_RGB.jpg'

    img_file1 = io.imread(file1)
    img_file2 =io.imread(file2)

    res = 0.3
    size = img_file1.shape[0]
    step = size//2

    common1 = img_file1[:, step:]
    common2 = img_file2[:, :step]

    plt.figure(figsize=(15, 8))
    plt.subplot(131)
    plt.imshow(common1)

    plt.subplot(132)
    plt.imshow(common2)

    plt.subplot(133)
    plt.imshow(common2-common1)

    plt.tight_layout()
    plt.show()

    # plt.savefig('/media/lab/Yang/data/synthetic_data/Airplanes/59_60common_change.jpg')

    plt.savefig('/media/lab/Yang/data/synthetic_data/Airplanes/75_76common_change.jpg')


def compare_images(two=False):
    syn_args = pps.get_syn_args()
    cities = ['barcelona', 'berlin', 'francisco', 'hexagon', 'radial', 'siena', 'spiral']
    streets = [200, 200, 200, 200, 200, 250, 130]
    syn_plane_img_anno_dir = '/media/lab/Yang/data/synthetic_data/Airplanes/{}'
    city = cities[0]
    sts = streets[0]

    step = syn_args.tile_size * syn_args.resolution

    if two:
        '''
        first --> 0 
        '''
        IMG_FORMAT = '.jpg'
        save_dir = os.path.join('/media/lab/Yang/data/synthetic_data/Airplanes/', 'check_rgb_gt0', city)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        tx_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture0'), '{}_{}_{}_images_step{}'.format('syn_texture0', city, sts, step))
        tx_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture0'), '{}_{}_{}_annos_step{}'.format('syn_texture0',  city, sts, step))

        clr_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color0'), '{}_{}_{}_images_step{}'.format('syn_color0', city, sts, step))
        clr_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color0'), '{}_{}_{}_annos_step{}'.format('syn_color0',  city, sts, step))
        tx_image_files = np.sort(glob(os.path.join(tx_img_dir, '*' + IMG_FORMAT)))
        tx_img_names = [os.path.basename(f) for f in tx_image_files]
        clr_image_files = np.sort(glob(os.path.join(clr_img_dir, '*' + IMG_FORMAT)))
        clr_img_names = [os.path.basename(f) for f in clr_image_files]

        tx_lbl_files = np.sort(glob(os.path.join(tx_lbl_dir, '*' + IMG_FORMAT)))
        # tx_lbl_names = [os.path.basename(f) for f in tx_lbl_files]
        clr_lbl_files = np.sort(glob(os.path.join(clr_lbl_dir, '*' + IMG_FORMAT)))
        # clr_lbl_names = [os.path.basename(f) for f in clr_lbl_files]

        for ix in range(len(tx_image_files)):
            file1 = tx_image_files[ix]
            file2 = os.path.join(clr_img_dir, clr_img_names[ix])

            img_file1 = io.imread(file1)
            img_file2 = io.imread(file2)

            file21 = tx_lbl_files[ix]
            file22 = clr_lbl_files[ix]

            img_file21 = io.imread(file21)
            img_file22 = io.imread(file22)

            fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

            axs[0, 0].imshow(img_file1)
            axs[0, 0].set_title('texture')
            axs[0, 1].imshow(img_file2)
            axs[0, 1].set_title('color')
            axs[1, 0].imshow(img_file21)
            axs[1, 1].imshow(img_file22)
            axs[1, 1].imshow(img_file21- img_file22)
            axs[1, 0].set_xlabel(tx_img_names[ix])
            axs[1, 2].set_xlabel(clr_img_names[ix])
            axs[1, 2].set_xlabel('left - right')
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, tx_img_names[ix].replace('texture_', '')))
            # plt.show()
    else:
        '''
        second 
        '''
        IMG_FORMAT = '.png'
        save_dir = os.path.join('/media/lab/Yang/data/synthetic_data/Airplanes/', 'check_rgb_gt', city)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        tx_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture'), '{}_{}_{}_images_step{}'.format('syn_texture', city, sts, step))
        tx_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture'), '{}_{}_{}_annos_step{}'.format('syn_texture',  city, sts, step))

        clr_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color'), '{}_{}_{}_images_step{}'.format('syn_color', city, sts, step))
        clr_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color'), '{}_{}_{}_annos_step{}'.format('syn_color',  city, sts, step))

        mx_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_mixed'), '{}_{}_{}_images_step{}'.format('syn_mixed', city, sts, step))
        mx_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_mixed'), '{}_{}_{}_annos_step{}'.format('syn_mixed',  city, sts, step))

        tx_image_files = np.sort(glob(os.path.join(tx_img_dir, '*' + IMG_FORMAT)))
        tx_img_names = [os.path.basename(f) for f in tx_image_files]
        clr_image_files = np.sort(glob(os.path.join(clr_img_dir, '*' + IMG_FORMAT)))
        clr_img_names = [os.path.basename(f) for f in clr_image_files]
        mx_image_files = np.sort(glob(os.path.join(mx_img_dir, '*' + IMG_FORMAT)))
        mx_img_names = [os.path.basename(f) for f in mx_image_files]

        tx_lbl_files = np.sort(glob(os.path.join(tx_lbl_dir, '*' + IMG_FORMAT)))
        # tx_lbl_names = [os.path.basename(f) for f in tx_lbl_files]
        clr_lbl_files = np.sort(glob(os.path.join(clr_lbl_dir, '*' + IMG_FORMAT)))
        # clr_lbl_names = [os.path.basename(f) for f in clr_lbl_files]
        mx_lbl_files = np.sort(glob(os.path.join(mx_lbl_dir, '*' + IMG_FORMAT)))
        # mx_lbl_names = [os.path.basename(f) for f in mx_lbl_files]

        for ix in range(len(tx_image_files)):
            file1 = tx_image_files[ix]
            file2 = os.path.join(clr_img_dir, clr_img_names[ix])
            file3 = os.path.join(mx_img_dir, mx_img_names[ix])

            img_file1 = io.imread(file1)
            img_file2 = io.imread(file2)
            img_file3 = io.imread(file3)

            file21 = tx_lbl_files[ix]
            file22 = clr_lbl_files[ix]
            file23 = mx_lbl_files[ix]

            img_file21 = io.imread(file21)
            img_file22 = io.imread(file22)
            img_file23 = io.imread(file23)

            # if np.all(img_file3==img_file2) and np.all(img_file2==img_file1):
            #     continue

            # print(tx_image_files[ix])

            fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

            axs[0, 0].imshow(img_file1)
            axs[0, 0].set_title('texture')
            axs[0, 1].imshow(img_file2)
            axs[0, 1].set_title('color')
            axs[0, 2].imshow(img_file3)
            axs[0, 2].set_title('mixed')
            axs[1, 0].imshow(img_file21)
            axs[1, 1].imshow(img_file22)
            axs[1, 2].imshow(img_file23)
            axs[1, 0].set_xlabel(tx_img_names[ix])
            axs[1, 1].set_xlabel(clr_img_names[ix])
            axs[1, 2].set_xlabel(mx_img_names[ix])
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, tx_img_names[ix].replace('texture_', '')))
            # plt.show()


def compare_images_after_combine(two=False):

    '''
    all images annos
    '''
    syn_args = pps.get_syn_args()
    syn_plane_img_anno_dir = '/media/lab/Yang/data/synthetic_data/Airplanes/{}'

    step = syn_args.tile_size * syn_args.resolution

    if two:
        '''
        first --> 0 
        '''
        IMG_FORMAT = '.jpg'
        save_dir = os.path.join('/media/lab/Yang/data/synthetic_data/Airplanes/', 'check_rgb_gt0', 'all')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            shutil.rmtree(save_dir)
            os.makedirs(save_dir)
        tx_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture0'), '{}_all_images_step{}'.format('syn_texture0', step))
        tx_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture0'), '{}_all_annos_step{}'.format('syn_texture0', step))

        clr_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color0'), '{}_all_images_step{}'.format('syn_color0', step))
        clr_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color0'), '{}_all_annos_step{}'.format('syn_color0', step))

        tx_image_files = np.sort(glob(os.path.join(tx_img_dir, '*' + IMG_FORMAT)))
        tx_img_names = [os.path.basename(f) for f in tx_image_files]
        clr_image_files = np.sort(glob(os.path.join(clr_img_dir, '*' + IMG_FORMAT)))
        clr_img_names = [os.path.basename(f) for f in clr_image_files]

        tx_lbl_files = np.sort(glob(os.path.join(tx_lbl_dir, '*' + IMG_FORMAT)))
        # tx_lbl_names = [os.path.basename(f) for f in tx_lbl_files]
        clr_lbl_files = np.sort(glob(os.path.join(clr_lbl_dir, '*' + IMG_FORMAT)))
        # clr_lbl_names = [os.path.basename(f) for f in clr_lbl_files]

        for ix in range(len(tx_image_files)):
            file1 = tx_image_files[ix]
            file2 = os.path.join(clr_img_dir, clr_img_names[ix])

            img_file1 = io.imread(file1)
            img_file2 = io.imread(file2)

            file21 = tx_lbl_files[ix]
            file22 = clr_lbl_files[ix]

            img_file21 = io.imread(file21)
            img_file22 = io.imread(file22)

            fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

            axs[0, 0].imshow(img_file1)
            axs[0, 0].set_title('texture')
            axs[0, 1].imshow(img_file2)
            axs[0, 1].set_title('color')
            axs[1, 0].imshow(img_file21)
            axs[1, 1].imshow(img_file22)
            axs[1, 0].set_xlabel(tx_img_names[ix])
            axs[1, 1].set_xlabel(clr_img_names[ix])
            axs[1, 2].imshow(img_file21- img_file22)
            axs[1, 2].set_xlabel('left - right')
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, tx_img_names[ix].replace('texture_', '')))
            # plt.show()
    else:
        IMG_FORMAT = '.png'
        save_dir = os.path.join('/media/lab/Yang/data/synthetic_data/Airplanes/', 'check_rgb_gt0', 'all')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        tx_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture'), '{}_all_images_step{}'.format('syn_texture', step))
        tx_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_texture'), '{}_all_annos_step{}'.format('syn_texture', step))

        clr_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color'), '{}_all_images_step{}'.format('syn_color', step))
        clr_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_color'), '{}_all_annos_step{}'.format('syn_color', step))

        mx_img_dir = os.path.join(syn_plane_img_anno_dir.format('syn_mixed'), '{}_all_images_step{}'.format('syn_mixed', step))
        mx_lbl_dir = os.path.join(syn_plane_img_anno_dir.format('syn_mixed'), '{}_all_annos_step{}'.format('syn_mixed', step))
        tx_image_files = np.sort(glob(os.path.join(tx_img_dir, '*' + IMG_FORMAT)))
        tx_img_names = [os.path.basename(f) for f in tx_image_files]
        clr_image_files = np.sort(glob(os.path.join(clr_img_dir, '*' + IMG_FORMAT)))
        clr_img_names = [os.path.basename(f) for f in clr_image_files]
        mx_image_files = np.sort(glob(os.path.join(mx_img_dir, '*' + IMG_FORMAT)))
        mx_img_names = [os.path.basename(f) for f in mx_image_files]

        tx_lbl_files = np.sort(glob(os.path.join(tx_lbl_dir, '*' + IMG_FORMAT)))
        # tx_lbl_names = [os.path.basename(f) for f in tx_lbl_files]
        clr_lbl_files = np.sort(glob(os.path.join(clr_lbl_dir, '*' + IMG_FORMAT)))
        # clr_lbl_names = [os.path.basename(f) for f in clr_lbl_files]
        mx_lbl_files = np.sort(glob(os.path.join(mx_lbl_dir, '*' + IMG_FORMAT)))
        # mx_lbl_names = [os.path.basename(f) for f in mx_lbl_files]
        for ix in range(len(tx_image_files)):
            file1 = tx_image_files[ix]
            file2 = os.path.join(clr_img_dir, clr_img_names[ix])
            file3 = os.path.join(mx_img_dir, mx_img_names[ix])

            img_file1 = io.imread(file1)
            img_file2 = io.imread(file2)
            img_file3 = io.imread(file3)

            file21 = tx_lbl_files[ix]
            file22 = clr_lbl_files[ix]
            file23 = mx_lbl_files[ix]

            img_file21 = io.imread(file21)
            img_file22 = io.imread(file22)
            img_file23 = io.imread(file23)

            fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

            axs[0, 0].imshow(img_file1)
            axs[0, 0].set_title('texture')
            axs[0, 1].imshow(img_file2)
            axs[0, 1].set_title('color')
            axs[0, 2].imshow(img_file3)
            axs[0, 2].set_title('mixed')
            axs[1, 0].imshow(img_file21)
            axs[1, 1].imshow(img_file22)
            axs[1, 2].imshow(img_file23)
            axs[1, 0].set_xlabel(tx_img_names[ix])
            axs[1, 1].set_xlabel(clr_img_names[ix])
            axs[1, 2].set_xlabel(mx_img_names[ix])
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, tx_img_names[ix].replace('texture_', '')))
            # plt.show()


if __name__ == "__main__":
    # display_type = ['syn_texture', 'syn_color', 'syn_mixed']
    # # display_type = ['syn_texture0', 'syn_color0']
    # city = ['barcelona', 'berlin', 'francisco', 'hexagon', 'radial', 'siena', 'spiral']
    # streets = [200, 200, 200, 200, 200, 250, 130]

    # two = True
    # compare_images_separate_cities(two)

    two = True
    compare_images_after_combine(two)
