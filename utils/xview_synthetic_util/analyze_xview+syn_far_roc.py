import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from ast import literal_eval
from PIL import ImageColor


def plot_roc(comments):
    lgd_font = "{'family': 'serif', 'weight': 'normal', 'size': 8}"
    tlt_font = "{'family': 'serif', 'weight': 'normal', 'size': 13}"
    sd = 17
#    ap_list = [50, 40, 20]
    apN = 50
    ehtypes = ['hard', 'easy']
    model_ids = [4, 1, 5, 5, 5]
    rare_classes = [1, 2, 3, 4, 5]
    marker_list = ['-^', '-v', '-<', '->', '-o', '-*']
    hyp_cmt = 'hgiou1_x4s4'
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

            save_name = 'ROC_xview+syn_RC{}_AP{}_{}.png'.format(rare_id, apN, ehtp)
            lix = cmt.rfind('_')
            lgd = 'xview+syn_{}_AP{}'.format(cmt[rix:lix], apN) # 'RC*_v*_AP*'
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
    comments = ['xview_syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynsigma_size_bias0.15_RC5_v43_color']
<<<<<<< HEAD
    plot_roc(comments)
=======
    for cmt in comments:
        plot_roc(cmt)
>>>>>>> master



















