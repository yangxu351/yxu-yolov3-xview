import matplotlib.pyplot as plt
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import pandas as pd
import numpy as np
from ast import literal_eval
from PIL import ImageColor


def dynamic_mu_size(base_range, pros, size_base, base_version, rare_id, save_dir, file_name):
    df_size = pd.DataFrame(columns=['Version', 'size_mean', 'size_std'], index=None)
    for ix in range(len(pros)):
        low = base_range[0] + pros[ix]*size_base
        high = base_range[1] + pros[ix]*size_base
        size_mean = (low+high)/2
        size_std = np.around(np.sqrt(np.power(high-low, 2)/12), decimals=2)
        # print('unif mean, std', unif_rgb_mean, unif_rgb_std)

        vix = ix + base_version
        df_size = df_size.append({'Version':vix, 'size_mean':size_mean, 'size_std':size_std}, ignore_index=True)
#    print(df_size)
    with pd.ExcelWriter(os.path.join(save_dir, file_name), mode='w') as writer:
        df_size.to_excel(writer, sheet_name='RC{}'.format(rare_id), index=False)


def plot_roc_of_dynamic_mu_size(comments, base_range, pros, size_base, base_version):
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
            dynamic_mu_size(base_range, pros, size_base,base_version, rare_id, save_dir, file_name='dynmu_size_RC{}.xlsx'.format(rare_id))

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

            ax_roc.plot(df_far_thres.loc[:], df_rec_thres.loc[:], marker_list[ix], linewidth=2.5, markersize=5, alpha=0.6)
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
    '''
    RC1 dynmu size
    '''
#    comments = []
#    base_range = [12.5, 14]
#    pros = [0, 1, 2, 3, 4, 5]
#    size_base = 5
#    base_version = 130
#    for ix, pro in enumerate(pros):
#       cmt = 'syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_dynmu_size_bias{}_RC1_v{}_color'.format(pro*size_base, ix+ base_version)
#       comments.append(cmt)
##    base_version = 110
##    for ix, pro in enumerate(pros):
##        cmt = 'syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_fixedangle_dynmu_size_bias{}_RC1_v{}_color'.format(pro*size_base, ix+ base_version)
##        comments.append(cmt)
#
#    plot_roc_of_dynamic_mu_size(comments, base_range, pros, size_base, base_version)

    '''
    RC2 dynmu size
    '''
    comments = []
    base_range = [37, 40]
    pros = [-2, -1, 0, 1, 1.5, 2]
    size_base = 10
    base_version = 23
    for ix, pro in enumerate(pros):
       cmt = 'syn_xview_bkg_px23whr3_xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_dynmu_size_bias{}_RC2_v{}_color'.format(pro*size_base, ix+ base_version)
       comments.append(cmt)
    plot_roc_of_dynamic_mu_size(comments, base_range, pros, size_base, base_version)


    '''
    RC3 dynmu size
    '''
    comments = []
    base_range = [22, 28]
    pros = [-1, 0, 1, 2, 3, 4]
    size_base = 7
    base_version = 23
    for ix, pro in enumerate(pros):
       cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_size_bias{}_RC3_v{}_color'.format(pro*size_base, ix+ base_version)
       comments.append(cmt)
    plot_roc_of_dynamic_mu_size(comments, base_range, pros, size_base, base_version)

    '''
    RC4 dynmu size
    '''
    comments = []
    base_range = [30, 32]
    pros = [-3, -2, -1, 0, 1, 2]
    size_base = 5
    base_version = 23
    for ix, pro in enumerate(pros):
       cmt = 'syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_size_bias{}_RC4_v{}_color'.format(pro*size_base, ix+ base_version)
       comments.append(cmt)
    plot_roc_of_dynamic_mu_size(comments, base_range, pros, size_base, base_version)

    '''
    RC5 dynmu size
    '''
    comments = []
    base_range = [7.5, 10]
    pros = [0, 1, 2, 3, 4, 5]
    size_base = 5
    base_version = 12
    for ix, pro in enumerate(pros):
       cmt = 'syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_size_bias{}_RC5_v{}_color'.format(pro*size_base, ix+ base_version)
       comments.append(cmt)
    plot_roc_of_dynamic_mu_size(comments, base_range, pros, size_base, base_version)
