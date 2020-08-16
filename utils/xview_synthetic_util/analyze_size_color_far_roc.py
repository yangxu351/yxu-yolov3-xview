import matplotlib.pyplot as plt
import os
import pandas as pd
# import sys
# sys.path.append('yxu-yolov3-xview')


if __name__ == '__main__':
    sd = 17
    apN = 50
    ehtypes = ['hard', 'easy']
    model_ids = [4, 1, 5, 5, 5]
    rare_classes = [1, 2, 3, 4, 5]
    hyp_cmt = 'hgiou1_1gpu_val_syn'
    far_thres = 3
    comments = []
    '''RC1'''
    # pros = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    # base_version = 50
    # for ix, pro in enumerate(pros):
    #     cmt = 'syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias{}_RC1_v{}_color'.format(
    #         pro, ix + base_version)
    #     comments.append(cmt)

    for ehtp in ehtypes:
        fig, ax_roc = plt.subplots(figsize=(10, 8))

        legends = []
        for ix, cmt in enumerate(comments):
            rix = cmt.finx('RC')
            rare_id = cmt[rix + 2]
            model_id = model_ids[rare_classes.index(rare_id)]
            ### baseline -- xview
            # if ix == 0:
            #     base_src_dir = 'result_output/1_cls/px23whr3_seed17/test_on_xview_hgiou1_1gpu_xview_only_m{}_rc{}_ap{}_{}'.format(model_id, rare_id, apN, ehtp)
            #     df_base_rec = pd.read_csv(os.path.join(base_src_dir, 'rec_list.txt'), header=None)
            #     df_base_far = pd.read_csv(os.path.join(base_src_dir, 'far_list.txt'), header=None)
            #     df_base_far_thres = df_base_far[df_base_far<=far_thres]
            #     df_base_far_thres = df_base_far_thres.dropna()
            #     df_base_rec_thres = df_base_rec.loc[:df_base_far_thres.shape[0]-1]
            #     ax_roc.plot(df_base_rec_thres.loc[:], df_base_far_thres.loc[:], '-o', markersize=3)
            #     lgd = 'xview_RC{}_{}'.format(rare_id, ehtp)
            #     legends.append(lgd)

            folder = 'test_on_xview_{}_m{}_rc{}_ap{}_{}'.format(hyp_cmt, model_id, rare_id, apN, ehtp)
            result_src_dir = 'result_output/1_cls/{}_seed{}/{}/'.format(cmt, sd, folder)
            bix = cmt.find('bias')
            save_dir = 'result_output/1_cls/{}_RC{}/'.format(cmt[:bix + 4], rare_id)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
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

            ax_roc.plot(df_rec_thres.loc[:], df_far_thres.loc[:], '-o', markersize=3)
            ax_roc.set_title('ROC of Last Epoch')
            ax_roc.set_xlabel('FAR')
            ax_roc.set_ylabel('Recall')
            ax_roc.set_ylim(-0.5, 1.1)
            ax_roc.set_xlim(-0.5,4)
            ax_roc.grid(True)
        fig.legend(legends)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, save_name), dpi=300)
        plt.close(fig)
