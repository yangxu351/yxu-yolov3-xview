import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from ast import literal_eval
from PIL import ImageColor

def hex_to_rgb(hexes):
    hex_list = [s for s in hexes.split(';')]
    r_list = []
    g_list = []
    b_list = []
    for hex in hex_list:
        (r, g, b) = ImageColor.getcolor(hex, "RGB")
        r_list.append(r)
        g_list.append(g)
        b_list.append(b)
    r_mean = np.round(np.mean(r_list)).astype(np.int)
    g_mean = np.round(np.mean(g_list)).astype(np.int)
    b_mean = np.round(np.mean(b_list)).astype(np.int)
    r_std = np.std(r_list)
    g_std = np.std(g_list)
    b_std = np.std(b_list)
    rgb_mean = np.round([r_mean, g_mean, b_mean])
    rgb_std = np.around([r_std, g_std, b_std], decimals=1)
    print('rgb mean', rgb_mean)
    print('rgb std', rgb_std)
    return rgb_mean, rgb_std


def dynamic_sigma_color(rgb_mean, pros, base_version, rare_id, save_dir, file_name):
    df_rgb = pd.DataFrame(columns=['Version', 'rgb_mean', 'rgb_std'], index=None)
    for ix, pro in enumerate(pros):
        bp = pro * rgb_mean
        low = np.clip(rgb_mean - bp, 0, 255)
        high = np.clip(rgb_mean + bp, 0, 255)
        print('low', low, 'high', high)
        unif_rgb_mean = np.round((low+high)/2)
        unif_rgb_std = np.around(np.sqrt(np.power(high-low,2)/12), decimals=2)
        print('unif mean, std', unif_rgb_mean, unif_rgb_std)

        vix = ix + base_version
        df_rgb = df_rgb.append({'Version':vix, 'rgb_mean':unif_rgb_mean, 'rgb_std':unif_rgb_std}, ignore_index=True)
    print('df_rgb', df_rgb)

    with pd.ExcelWriter(os.path.join(save_dir, file_name), mode='w') as writer:
        df_rgb.to_excel(writer, sheet_name='RC{}'.format(rare_id), index=False)


def plot_roc_of_dynamic_sigma_color(comments, pros, hexes):
    lgd_font = "{'family': 'serif', 'weight': 'normal', 'size': 8}"
    tlt_font = "{'family': 'serif', 'weight': 'normal', 'size': 13}"
    sd = 17
    apN = 50
    ehtypes = ['hard', 'easy']
    model_ids = [4, 1, 5, 5, 5]
    rare_classes = [1, 2, 3, 4, 5]
    marker_list = ['-o', '-^', '-v', '-<', '->']
    hyp_cmt = 'hgiou1_1gpu_val_syn'
    far_thres = 3
    for ehtp in ehtypes:
        fig, ax_roc = plt.subplots(1, 1)  # figsize=(10, 8)
        yticks = [0]
        legends = []
        for ix, cmt in enumerate(comments):
            rix = cmt.find('RC')
            rare_id = int(cmt[rix + 2])
            bix = cmt.find('bias')
            model_id = model_ids[rare_classes.index(rare_id)]
            folder = 'test_on_xview_{}_m{}_rc{}_ap{}_{}'.format(hyp_cmt, model_id, rare_id, apN, ehtp)
            result_src_dir = '/data/users/yang/code/result_output/1_cls/{}_seed{}/{}/'.format(cmt, sd, folder)
            save_dir = '/data/users/yang/code/result_output/1_cls/{}_RC{}/'.format(cmt[:bix + 4], rare_id)
            # result_src_dir = '/media/lab/Yang/code/results_groot/1_cls/{}_seed{}/{}/'.format(cmt, sd, folder)
            # save_dir = '/media/lab/Yang/code/results_groot/1_cls/{}_RC{}/'.format(cmt[:bix + 4], rare_id)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            rgb_mean, _ = hex_to_rgb(hexes)
            dynamic_sigma_color(rgb_mean, pros, base_version, rare_id, save_dir, file_name='dynsigma_color_RC{}.xlsx'.format(rare_id))

            dix = cmt.find('dyn')
            save_name = 'ROC_{}_RC{}_{}.png'.format(cmt[dix:bix + 4], rare_id, ehtp)
            lix = cmt.rfind('_')
            lgd = 'syn_{}'.format(cmt[rix:lix]) # 'RC*_v*'
            legends.append(lgd)

            df_rec = pd.read_csv(os.path.join(result_src_dir, 'rec_list.txt'), header=None)
            df_far = pd.read_csv(os.path.join(result_src_dir, 'far_list.txt'), header=None)
            df_far_thres = df_far[df_far<=far_thres]
            df_far_thres = df_far_thres.dropna()
            df_rec_thres = df_rec.loc[:df_far_thres.shape[0]-1]
            yticks.extend(df_rec_thres.loc[:,0].tolist())

            ax_roc.plot(df_far_thres.loc[:], df_rec_thres.loc[:], marker_list[ix], linewidth=2.5, markersize=5, alpha=0.6)
            ax_roc.set_title('ROC of RC{} {}'.format(rare_id, ehtp), literal_eval(tlt_font))
            ax_roc.set_xlabel('FAR', literal_eval(tlt_font))
            ax_roc.set_ylabel('Recall', literal_eval(tlt_font))
            # ax_roc.set_ylim(-0.05, 1.05)
            ax_roc.set_xlim(-0.05, 3.05)
            ax_roc.grid(True)

        fig.legend(legends, prop=literal_eval(lgd_font), loc='upper right')
        yticks.append(1)
        yticks = list(dict.fromkeys(yticks))
        print('yticks', yticks)
        plt.yticks(yticks)
        plt.ylim(0.05, 1.05)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, save_name), dpi=300)
        plt.close(fig)


if __name__ == '__main__':

    '''RC1  dynsigma color
     '''
    comments = []
    pros = [0, 0.2, 0.4, 0.6, 0.8]
    hexes = "#685C45;#786755;#897954;#8D7C6A;#7B6A59;#857861"
    base_version = 60
    for ix, pro in enumerate(pros):
        cmt = 'syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_dynsigma_color_bias{}_RC1_v{}_color'.format(
            pro, ix + base_version)
        comments.append(cmt)
    plot_roc_of_dynamic_sigma_color(comments, pros, hexes)
    ##########################################

    '''RC2  dynsigma color
     '''
    comments = []
    pros = [0, 0.2, 0.4, 0.6, 0.8]
    hexes = "#E5DAD8;#F5EAE1;#E6E5E6;#EBE6E1;#FFFEFF;#FEF9F7"
    base_version = 40
    for ix, pro in enumerate(pros):
        cmt = 'syn_xview_bkg_px23whr3_xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_dynsigma_color_bias{}_RC2_v{}_color'.format(
            pro, ix + base_version)
        comments.append(cmt)
    plot_roc_of_dynamic_sigma_color(comments, pros, hexes)
    ##########################################

    '''RC3  dynsigma color
     '''
    comments = []
    pros = [0, 0.2, 0.4, 0.6, 0.8]
    hexes = "#FFFFF9;#FFFFEF;#FFFFFB;#F9F5E6;#F7F6F0;#EFEEE7"
    base_version = 40
    for ix, pro in enumerate(pros):
        cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynsigma_color_bias{}_RC3_v{}_color'.format(
            pro, ix + base_version)
        comments.append(cmt)
    plot_roc_of_dynamic_sigma_color(comments, pros, hexes)
    ##########################################

    '''RC4  dynsigma color
     '''
    comments = []
    pros = [0, 0.2, 0.4, 0.6, 0.8]
    hexes = "#4C4B4E;#454545;#4A4744;#3D382C;#49483B;#453F37;#403B31;#504E41"
    base_version = 40
    for ix, pro in enumerate(pros):
        cmt = 'syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynsigma_color_bias{}_RC4_v{}_color'.format(
            pro, ix + base_version)
        comments.append(cmt)
    plot_roc_of_dynamic_sigma_color(comments, pros, hexes)
    ##########################################

    '''RC5  dynsigma color
     '''
    comments = []
    pros = [0, 0.2, 0.4, 0.6, 0.8]
    hexes = "#C28634;#9C6932;#AA752B;#D89B43;#CD9644;#846830"
    base_version = 30
    for ix, pro in enumerate(pros):
        cmt = 'syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynsigma_color_bias{}_RC5_v{}_color'.format(
            pro, ix + base_version)
        comments.append(cmt)
    plot_roc_of_dynamic_sigma_color(comments, pros, hexes)
    ##########################################

















    ### baseline -- xview
    # if ix == 0:
    #     base_src_dir = '../../result_output/1_cls/px23whr3_seed17/test_on_xview_hgiou1_1gpu_xview_only_m{}_rc{}_ap{}_{}'.format(model_id, rare_id, apN, ehtp)
    #     df_base_rec = pd.read_csv(os.path.join(base_src_dir, 'rec_list.txt'), header=None)
    #     df_base_far = pd.read_csv(os.path.join(base_src_dir, 'far_list.txt'), header=None)
    #     df_base_far_thres = df_base_far[df_base_far<=far_thres]
    #     df_base_far_thres = df_base_far_thres.dropna()
    #     df_base_rec_thres = df_base_rec.loc[:df_base_far_thres.shape[0]-1]
    #     ax_roc.plot(df_base_far_thres.loc[:], df_base_rec_thres.loc[:], '-o', linewidth=2.5, markersize=5)
    #     lgd = 'xview_RC{}_{}'.format(rare_id, ehtp)
    #     legends.append(lgd)

