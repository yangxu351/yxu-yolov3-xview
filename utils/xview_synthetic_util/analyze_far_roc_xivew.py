import os
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval

if __name__ == '__main__':
    lgd_font = "{'family': 'serif', 'weight': 'normal', 'size': 8}"
    tlt_font = "{'family': 'serif', 'weight': 'normal', 'size': 13}"
    ## baseline -- xview
    save_dir = '/media/lab/Yang/code/results_groot/1_cls_pre_rec/xview_only_RC1-5_ROC/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    sd = 17
    ap_list = [20, 40, 50]  # 20 # 50
    ehtypes = ['hard', 'easy']
    far_thres = 3
    model_ids = [4, 1, 5, 5, 5]
    rare_classes = [1, 2, 3, 4, 5]
    marker_list = ['-^', '-v', '-<', '->', '-o', '-*']
    for apN in ap_list:
        for ehtp in ehtypes:
            fig, ax_roc = plt.subplots(1, 1)
            legends = []
            yticks = [0]
            for ix in range(len(model_ids)):
                model_id = model_ids[ix]
                rare_id = rare_classes[ix]
                base_src_dir = '../../result_output/1_cls/px23whr3_seed17/test_on_xview_hgiou1_1gpu_xview_only_m{}_rc{}_ap{}_{}'.format(model_id, rare_id, apN, ehtp)
                df_base_rec = pd.read_csv(os.path.join(base_src_dir, 'rec_list.txt'), header=None)
                df_base_far = pd.read_csv(os.path.join(base_src_dir, 'far_list.txt'), header=None)
                df_base_far_thres = df_base_far[df_base_far<=far_thres]
                # df_base_far_thres = df_base_far
                df_base_far_thres = df_base_far_thres.dropna()
                df_base_rec_thres = df_base_rec.loc[:df_base_far_thres.shape[0]-1]
                yticks.extend(df_base_rec_thres.loc[:,0].tolist())

                ax_roc.plot(df_base_far_thres.loc[:], df_base_rec_thres.loc[:], marker_list[ix], linewidth=2.5, markersize=5)
                lgd = 'xview_RC{}_AP{}_{}'.format(rare_id, apN, ehtp)
                ax_roc.set_xlim(-0.05, 3.05)
                ax_roc.set_title('ROC of xview {}'.format(ehtp), literal_eval(tlt_font))
                ax_roc.set_xlabel('FAR', literal_eval(tlt_font))
                ax_roc.set_ylabel('Recall', literal_eval(tlt_font))
                ax_roc.grid(True)
                legends.append(lgd)
            fig.legend(legends, prop=literal_eval(lgd_font))
            yticks.append(1)
            yticks = list(dict.fromkeys(yticks))
            print('yticks', yticks)
            plt.yticks(yticks)
            plt.ylim(0.05, 1.05)
            plt.tight_layout()
            save_name = 'xview_only_RC1-5_ROC_AP{}_{}.png'.format(apN, ehtp)
            fig.savefig(os.path.join(save_dir, save_name), dpi=300)
            plt.close(fig)
