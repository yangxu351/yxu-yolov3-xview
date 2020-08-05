import matplotlib.pyplot as plt
import os
import pandas as pd

if __name__ == '__main__':

#    comments = ["syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_scatter_gauss_40_bias0_model5_v1_color", "syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_scatter_gauss_40_color_bias76.5_model5_v4_color", "syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_scatter_gauss_40_color_bias178.5_model5_v8_color", "syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_scatter_gauss_40_color_bias255.0_model5_v11_color"]
#    model_id = 5
#    rare_cls = 3
    
#    comments = ["syn_xview_bkg_px23whr3_xbsw_xwing_scatter_gauss_30_bias0_model1_v1_color", "syn_xview_bkg_px23whr3_xbsw_xwing_scatter_gauss_30_color_bias76.5_model1_v4_color", "syn_xview_bkg_px23whr3_xbsw_xwing_scatter_gauss_30_color_bias178.5_model1_v8_color", "syn_xview_bkg_px23whr3_xbsw_xwing_scatter_gauss_30_color_bias255.0_model1_v11_color"]
#    model_id = 1
#    rare_cls = 2

    comments = ["syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_rnd_color_bias0_model4_v21_color", "syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_rnd_color_bias127.5_model4_v26_color", "syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_rnd_color_bias178.5_model4_v28_color", "syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_rnd_color_bias229.5_model4_v30_color"]
    model_id = 4
    rare_cls = 1

    far_thres = 3
    apN = 20
    hyp_cmt = 'hgiou1_1gpu_val_syn'

    px_thres = 23
    whr_thres = 3 # 4
    sd = 17
    eh_types = ['hard', 'easy']
    X = [x for x in range(5, 220, 5)]
    for typ in eh_types:
        fig, ((ax_r, ax_ap), (ax_f1, ax_roc)) = plt.subplots(2, 2, figsize=(12, 10))
        legends = []
        for cmt in comments:
            if 'bias0' not in cmt:
                folder_name = '{}_{}'.format(cmt[:cmt.find('bias')+4], cmt[cmt.find('mo'):cmt.find('_v')])
                save_dir = 'result_output/1_cls/{}/'.format(folder_name)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                save_img_name = '{}_iou{}_bkups_{}.png'.format(folder_name, apN, typ)
            far_dir = 'result_output/1_cls/{}_seed{}/{}/'.format(cmt, sd, 'test_on_xview_{}_m{}_rc{}_{}_iou{}_last'.format(hyp_cmt, model_id, rare_cls, typ, apN))
#            print('far_dir', far_dir)
            df_rec = pd.read_csv(os.path.join(far_dir, 'rec_list.txt'), header=None)         
            df_far = pd.read_csv(os.path.join(far_dir, 'far_list.txt'), header=None)   
            df_far_thres = df_far[df_far<=far_thres]
            df_far_thres = df_far_thres.dropna()
            df_rec_thres = df_rec.loc[:df_far_thres.shape[0]-1]         
            result_dir = 'result_output/1_cls/{}_seed{}/'.format(cmt, sd)
            bkup_file_name = '{}_iou{}_bkups_{}.txt'.format(cmt, apN, typ)
            df_bkup = pd.read_csv(os.path.join(result_dir, bkup_file_name), header=None, sep=' ')
            ### 'Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.2', 'F1'
            legends.append(cmt[cmt.find('model'):])
            
            ax_r.plot(X, df_bkup.loc[:, 4], '-o', markersize=3)
            ax_r.set_title('Recall')
            ax_r.set_xlabel('epochs')
            ax_r.set_ylabel('Recall')
            ax_r.set_ylim(0, 1)
            ax_r.grid(True)
            ax_ap.plot(X, df_bkup.loc[:, 5], '-o', markersize=3)
            ax_ap.set_title('AP@0.2')
            ax_ap.set_xlabel('epochs')
            ax_ap.set_ylabel('AP@0.2')
            ax_ap.set_ylim(0, 1)
            ax_ap.grid(True)
            ax_f1.plot(X, df_bkup.loc[:, 6], '-o', markersize=3)
            ax_f1.set_title('F1')
            ax_f1.set_xlabel('epochs')
            ax_f1.set_ylabel('F1')
            ax_f1.set_ylim(0, 1)
            ax_f1.grid(True)
            
            ax_roc.plot(df_far_thres.loc[:], df_rec_thres.loc[:], '-o', markersize=4, alpha=0.5)
            ax_roc.set_title('ROC of Last Epoch')
            ax_roc.set_xlabel('FAR')
            ax_roc.set_ylabel('Recall')
            ax_roc.set_ylim(0, 1.1)
            ax_roc.set_xlim(-0.1, 3)
            ax_roc.grid(True)
        fig.legend(legends)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, save_img_name), dpi=300)
        plt.close(fig)
