import matplotlib.pyplot as plt
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
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


def dynamic_mu_color(rgb_mean, left_bias, right_bias,base_version, rare_id, save_dir, file_name):
    df_rgb = pd.DataFrame(columns=['Version', 'rgb_mean', 'rgb_std'], index=None)
    for ix in range(len(left_bias)):
        lbp = left_bias[ix] * 25.5
        rbp = right_bias[ix] * 25.5
        # print('lbp, rbp', lbp, rbp)
        low = np.clip(rgb_mean + lbp, 0, 255)
        high = np.clip(rgb_mean + rbp, 0, 255)
        # print('low', low, 'high', high)
        unif_rgb_mean = np.round((low+high)/2)
        unif_rgb_std = np.around(np.sqrt(np.power(high-low,2)/12), decimals=2)
        # print('unif mean, std', unif_rgb_mean, unif_rgb_std)

        vix = ix + base_version
        df_rgb = df_rgb.append({'Version':vix, 'rgb_mean':unif_rgb_mean, 'rgb_std':unif_rgb_std}, ignore_index=True)
    print('df_rgb', df_rgb)
    with pd.ExcelWriter(os.path.join(save_dir, file_name), mode='w') as writer:
        df_rgb.to_excel(writer, sheet_name='RC{}'.format(rare_id), index=False)


def plot_roc_of_dynamic_mu_color(comments, left_bias, right_bias, hexes):
    lgd_font = "{'family': 'serif', 'weight': 'normal', 'size': 8}"
    tlt_font = "{'family': 'serif', 'weight': 'normal', 'size': 13}"
    sd = 17
#    ap_list = [50, 40, 20]
    apN = 50
    ehtypes = ['hard', 'easy']
    model_ids = [4, 1, 5, 5, 5]
    rare_classes = [1, 2, 3, 4, 5]
    marker_list = ['-^', '-v', '-<', '->', '-o', '-*']
    hyp_cmt = 'hgiou1_1gpu_val_syn'
    far_thres = 3
    for ehtp in ehtypes:
#        for apN in ap_list:
#            df_roc = pd.DataFrame(columns=["Version", "Pd(FAR=0.5)", "Pd(FAR=1)"])
        fig, ax_roc = plt.subplots(1, 1)  # figsize=(10, 8)
        yticks = [0]
        legends = []
        for ix, cmt in enumerate(comments):
            rix = cmt.find('RC')
            rare_id = int(cmt[rix + 2])
            bix = cmt.find('bias')
            rcinx = rare_classes.index(rare_id)
            model_id = model_ids[rcinx]
            folder = 'test_on_xview_{}_m{}_rc{}_ap{}_{}'.format(hyp_cmt, model_id, rare_id, apN, ehtp)
            result_src_dir = '/data/users/yang/code/yxu-yolov3-xview/result_output/1_cls/{}_seed{}/{}/'.format(cmt, sd, folder)
            save_dir = '/data/users/yang/code/yxu-yolov3-xview/result_output/1_cls/{}_RC{}/'.format(cmt[:bix + 4], rare_id)
            # result_src_dir = '/media/lab/Yang/code/results_groot/1_cls/{}_seed{}/{}/'.format(cmt, sd, folder)
            # save_dir = '/media/lab/Yang/code/results_groot/1_cls/{}_RC{}/'.format(cmt[:bix + 4], rare_id)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            rgb_mean, _ = hex_to_rgb(hexes)
            dynamic_mu_color(rgb_mean, left_bias, right_bias, base_version, rare_id, save_dir, file_name='dynmu_color_RC{}.xlsx'.format(rare_id))

            dix = cmt.find('dyn')
            save_name = 'ROC_{}_RC{}_AP{}_{}.png'.format(cmt[dix:bix + 4], rare_id, apN, ehtp)
            lix = cmt.rfind('_')
            lgd = 'syn_{}_AP{}'.format(cmt[rix:lix], apN) # 'RC*_v*_AP*'
            legends.append(lgd)

            df_rec = pd.read_csv(os.path.join(result_src_dir, 'rec_list.txt'), header=None)
            df_far = pd.read_csv(os.path.join(result_src_dir, 'far_list.txt'), header=None)
            df_far_thres = df_far[df_far<=far_thres]
            df_far_thres = df_far_thres.dropna()
            df_rec_thres = df_rec.loc[:df_far_thres.shape[0]-1]
            yticks.extend(df_rec_thres.loc[:,0].tolist())

            ax_roc.plot(df_far_thres.loc[:], df_rec_thres.loc[:],  marker_list[ix], linewidth=2.5, markersize=5, alpha=0.6)
            ax_roc.set_title('ROC of RC{} {}'.format(rare_id, ehtp), literal_eval(tlt_font))
            ax_roc.set_xlabel('FAR', literal_eval(tlt_font))
            ax_roc.set_ylabel('Recall', literal_eval(tlt_font))
            # ax_roc.set_ylim(-0.05, 1.05)
            ax_roc.set_xlim(-0.05, 3.05)
            ax_roc.grid(True)

#                df_roc.at[ix, "Version"] = base_version + ix
#                idx5_mx = df_far[df_far>=0.5].dropna()
#                idx5_mx = idx5_mx.idxmin()[0]
#                idx5_mn = idx5_mx - 1
#                pd_5 = df_rec_thres.loc[idx5_mn, 0]
#                idx1_mx = df_far[df_far>=1].dropna()
#                idx1_mx = idx1_mx.idxmin()[0]
#                idx1_mn = idx1_mx - 1
#                pd_1 = df_rec_thres.loc[idx1_mn, 0]
#                df_roc.at[ix, "Pd(FAR=0.5)"] = pd_5
#                df_roc.at[ix, "Pd(FAR=1)"] = pd_1

        fig.legend(legends, prop=literal_eval(lgd_font), loc='upper right')
        yticks.append(1)
        yticks = list(dict.fromkeys(yticks))
        print('yticks', yticks)
        plt.yticks(yticks)
        plt.ylim(0.05, 1.05)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, save_name), dpi=300)
        plt.close(fig)
            
#            csv_name = 'Pd_{}_RC{}_{}.xlsx'.format(cmt[dix:bix + 4], rare_id, ehtp)
#            if os.path.exists(os.path.join(save_dir, csv_name)):
#                mode = 'a'
#            else:
#                mode = 'w'
#            with pd.ExcelWriter(os.path.join(save_dir, csv_name), mode=mode) as writer:
#                df_roc.to_excel(writer, sheet_name='AP{}'.format(apN), index=False) # 

if __name__ == '__main__':

    '''RC1  dynmu color
     mu color = [126 111  88]
     std = [12.4 11.5 11.5]
     '''
#    comments = []
#    left_bias = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
#    right_bias = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
#    hexes = "#685C45;#786755;#897954;#8D7C6A;#7B6A59;#857861"
#    base_version = 50
#    for ix, pro in enumerate(left_bias):
#        cmt = 'syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias{}_RC1_v{}_color'.format(
#            pro, ix + base_version)
#        comments.append(cmt)
##    base_version = 90
##    for ix, pro in enumerate(left_bias):
##        cmt = 'syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_fixedangle_dynmu_color_bias{}_RC1_v{}_color'.format(
##            pro, ix + base_version)
##        comments.append(cmt)
##
##    plot_roc_of_dynamic_mu_color(comments, left_bias, right_bias, hexes)
#    ##########################################
#
#    '''RC2  dynmu color
#     mu color = [241 236 233]
#     std = [10.7 12.3 13.6]
#     '''
#    comments = []
#    left_bias = [-0.5, -1.5, -2.5, -3.5, -4.5, -5.5]
#    right_bias = [0.5, -0.5, -1.5, -2.5, -3.5, -4.5]
#    hexes = "#E5DAD8;#F5EAE1;#E6E5E6;#EBE6E1;#FFFEFF;#FEF9F7"
#    base_version = 30
#    for ix, pro in enumerate(right_bias):
#        cmt = 'syn_xview_bkg_px23whr3_xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias{}_RC2_v{}_color'.format(
#            pro, ix + base_version)
#        comments.append(cmt)
#    plot_roc_of_dynamic_mu_color(comments, left_bias, right_bias, hexes)
    ##########################################

    '''RC3  dynmu color
     mu color = [250 249 240]
     std = [5.9 6.5 8. ]
     '''
    comments = []
    left_bias = [-0.5, -1.5, -2.5, -3.5, -4.5, -5.5]
    right_bias = [0.5, -0.5, -1.5, -2.5, -3.5, -4.5]
    hexes = "#FFFFF9;#FFFFEF;#FFFFFB;#F9F5E6;#F7F6F0;#EFEEE7"
    base_version = 30
    for ix, pro in enumerate(right_bias):
        cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias{}_RC3_v{}_color'.format(
            pro, ix + base_version)
        comments.append(cmt)
    plot_roc_of_dynamic_mu_color(comments, left_bias, right_bias, hexes)
#    ##########################################

    '''RC4  dynmu color
     mu color = [71 68 61]
     std = [ 5.9  7.3 10.6]
     '''
#    comments = []
#    left_bias = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
#    right_bias = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
#    hexes = "#4C4B4E;#454545;#4A4744;#3D382C;#49483B;#453F37;#403B31;#504E41"
#    base_version = 30 
#    for ix, pro in enumerate(left_bias):
#        cmt = 'syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias{}_RC4_v{}_color'.format(
#            pro, ix + base_version)
#        comments.append(cmt)
#    plot_roc_of_dynamic_mu_color(comments, left_bias, right_bias, hexes)
    ##########################################

    '''RC5  dynmu color
     mu color = [179 128  55]
     std = [29.1 20.3  9.5]
     '''
#    comments = []
#    left_bias = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
#    right_bias = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
#    hexes = "#C28634;#9C6932;#AA752B;#D89B43;#CD9644;#846830"
#    base_version = 20
#    for ix, pro in enumerate(left_bias):
#        cmt = 'syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias{}_RC5_v{}_color'.format(
#            pro, ix + base_version)
#        comments.append(cmt)
#    plot_roc_of_dynamic_mu_color(comments, left_bias, right_bias, hexes)
    ##########################################



















