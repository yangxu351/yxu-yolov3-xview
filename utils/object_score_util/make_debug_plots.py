import os
import numpy as np
import toolman as tm
import matplotlib.pyplot as plt
from tqdm import tqdm


# Settings
img_dir = r'/hdd/Bohao/figs'


def get_histogram(img_files, annos_files, progress=False):
    """
    Get the histogram of given list of images
    :param img_files: list of images, could be file names or numpy arrays
    :param progress: if True, will show a progress bar
    :return: a numpy array of size (3, 256) where each row represents histogram of certain color channel
    """
    hist = np.zeros((3, 256))
    if progress:
        pbar = tqdm(zip(img_files, annos_files), total=len(img_files))
    else:
        pbar = img_files
    for img_file, anno_file in pbar:
        if isinstance(img_file, str):
            img = tm.misc_utils.load_file(img_file)
        else:
            img = img_file

        anno = tm.misc_utils.load_files(anno_file)
        img[anno[:, :, 0] != 0] = 0

        for channel in range(3):
            img_hist, _ = np.histogram(img[:, :, channel].flatten(), bins=np.arange(0, 257))
            hist[channel, :] += img_hist
    return hist[:, 1:]


def plot_hist(hist, smooth=False):
    import scipy.signal
    color_list = ['r', 'g', 'b']
    for c in range(3):
        if smooth:
            plt.plot(scipy.signal.savgol_filter(hist[c, :], 11, 2), color_list[c])
        else:
            plt.plot(hist[c, :], color_list[c])


def make_hist_plots(rc_class=4, parent_dir='/media/lab/Seagate Expansion Drive/syn'):
    sigs = [0, 15, 30, 45, 60]
    if rc_class == 1:
        fold_name = 'syn_xview_bkg_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_ssig0_color_bias{}_RC1_v5{}'
    elif rc_class == 2:
        fold_name = 'syn_xview_bkg_xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_ssig0_color_bias{}_RC2_v5{}'
    elif rc_class == 3:
        fold_name = 'syn_xview_bkg_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_bias{}_RC3_v5{}'
    elif rc_class == 4:
        fold_name = 'syn_xview_bkg_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_bias{}_RC4_v5{}'
    elif rc_class == 5:
        fold_name = 'syn_xview_bkg_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.03_color_bias{}_RC5_v5{}'

    sig_imgs = []
    for sig_cnt, sig in enumerate(sigs):
        sig_imgs.append(tm.misc_utils.get_files(os.path.join(parent_dir, fold_name.format(sig, sig_cnt),
                                                             'color_all_images_step182.4'), '*.png'))
    annos_imgs = tm.misc_utils.get_files(os.path.join(parent_dir, fold_name.format(0, 0),
                                                             'color_all_annos_step182.4'), '*.png')

    assert len(sig_imgs[0]) == len(sig_imgs[1]) == len(sig_imgs[2]) == len(sig_imgs[3]) == len(sig_imgs[4]) == len(annos_imgs)

    plt.figure(figsize=(12, 8))
    for sig_cnt, sig in enumerate(sigs):
        plt.subplot(2, 3, sig_cnt+1)
        plot_hist(get_histogram(sig_imgs[sig_cnt], annos_imgs, progress=True), smooth=False)
        plt.title(f'RC{rc_class}_$\sigma_c={sig}$')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f'rc{rc_class}_hist_cmp.png'))
    plt.close()


def cmp_annotation(parent_dir=r'/media/lab/Seagate Expansion Drive/syn_gt_box'):
    sig0_dir = 'syn_xview_bkg_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_bias0_RC4_v50_gt_bbox'
    sig15_dir = 'syn_xview_bkg_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_bias15_RC4_v51_gt_bbox'
    sig0_imgs = tm.misc_utils.get_files(os.path.join(parent_dir, sig0_dir,
                                                     'minr100_linkr15_px23whr3_color_all_images_with_bbox_step182.4'),
                                        '*.png')
    sig15_imgs = tm.misc_utils.get_files(os.path.join(parent_dir, sig15_dir,
                                                      'minr100_linkr15_px23whr3_color_all_images_with_bbox_step182.4'),
                                         '*.png')

    assert len(sig0_imgs) == len(sig15_imgs)

    for sig0_img, sig15_img in zip(sig0_imgs, sig15_imgs):
        sig0, sig15 = tm.misc_utils.load_files([sig0_img, sig15_img])

        tm.vis_utils.compare_figures([sig0, sig15,
                                      np.abs(sig0.astype(np.float32)-sig15.astype(np.float32)).astype(np.uint8)*20],
                                     (1, 3), (12, 4),
                                     title_list=['$\sigma_c=0$', '$\sigma_c=15$', 'Diff'], show_fig=False)
        plt.savefig(os.path.join(img_dir, 'annos_cmp', os.path.basename(sig0_img)))
        plt.close()



if __name__ == '__main__':
    make_hist_plots(1)
    make_hist_plots(2)
    make_hist_plots(3)
    make_hist_plots(5)
