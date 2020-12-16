import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ast import literal_eval
import numpy as np


def plot_ROC_with_far_less_than_3(far_thres):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # for apN in ap_list:
    for ehtp in ehtypes:
        fig, ax_roc = plt.subplots(1, 1)
        legends = []
        yticks = [0]
        for ix in range(len(model_ids)):
            model_id = model_ids[ix]
            rare_id = rare_classes[ix]
            base_src_dir = '/data/users/yang/code/yxu-yolov3-xview/result_output/1_cls/px23whr3_seed17/test_on_xview_hgiou1_1gpu_xview_only_m{}_rc{}_ap{}_{}'.format(model_id, rare_id, apN, ehtp)
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


def plot_syn_xview_rc_ROC_with_far_tick(xbs_list, xview_src_path, syn_cmts, syn_src_path, save_path, xtick=[0.1, 1, 10]):
    
    eht = 'easy'
    fig, ax_roc = plt.subplots(1, 1)
    syn_seeds = [0, 1, 2]
    xview_seeds = [0, 1]
    log_xtick = [np.log10(x) for x in xtick]
    for ix, xbs in enumerate(xbs_list):
        rc_id = rare_classes[ix]
        syn_rec_avg_seeds = []
        xview_rec_avg_seeds = []
        for seed in syn_seeds:
            syn_src_dir = syn_src_path.format(syn_cmts[ix], rc_id, apN, seed)
            df_syn_rec = pd.read_csv(os.path.join(syn_src_dir, 'rec_list.txt'), header=None)
            df_syn_far = pd.read_csv(os.path.join(syn_src_dir, 'far_list.txt'), header=None)
            df_syn_far_0 = df_syn_far[df_syn_far<=xtick[0]]
            df_syn_far_0 = df_syn_far_0.dropna()
            syn_rec_0 = df_syn_rec.loc[df_syn_far_0.shape[0]-1]
            
            df_syn_far_1 = df_syn_far[df_syn_far<=xtick[1]]
            df_syn_far_1 = df_syn_far_1.dropna()
            syn_rec_1 = df_syn_rec.loc[df_syn_far_1.shape[0]-1]
            
            df_syn_far_2 = df_syn_far[df_syn_far<=xtick[2]]
            df_syn_far_2 = df_syn_far_2.dropna()
            syn_rec_2 = df_syn_rec.loc[df_syn_far_2.shape[0]-1]
            syn_rec_avg_seeds.append([syn_rec_0, syn_rec_1, syn_rec_2])
        syn_rec_avg_seeds = np.array(syn_rec_avg_seeds)
        syn_rec_avg = np.mean(syn_rec_avg_seeds, axis=0)
        lgd = 'syn_RC{}'.format(rc_id)
        ax_roc.plot(log_xtick, syn_rec_avg, linestyle='-', marker=marker_list[ix], color=color_list[ix], linewidth=1, markersize=2.5, label=lgd)
         
        for seed in xview_seeds:
            xview_src_dir = xview_src_path.format(rc_id, xbs, 8-xbs, apN, seed)    
            df_base_rec = pd.read_csv(os.path.join(xview_src_dir, 'rec_list.txt'), header=None)
            df_base_far = pd.read_csv(os.path.join(xview_src_dir, 'far_list.txt'), header=None)
            df_base_far_0 = df_base_far[df_base_far<=xtick[0]]
            df_base_far_0 = df_base_far_0.dropna()
            xview_rec_0 = df_base_rec.loc[df_base_far_0.shape[0]-1]
            
            df_base_far_1 = df_base_far[df_base_far<=xtick[1]]
            df_base_far_1 = df_base_far_1.dropna()
            xview_rec_1 = df_base_rec.loc[df_base_far_1.shape[0]-1]
            
            df_base_far_2 = df_base_far[df_base_far<=xtick[2]]
            df_base_far_2 = df_base_far_2.dropna()
            xview_rec_2 = df_base_rec.loc[df_base_far_2.shape[0]-1]
    
            xview_rec_avg_seeds.append([xview_rec_0, xview_rec_1, xview_rec_2])
        xview_rec_avg_seeds = np.array(xview_rec_avg_seeds)
        xview_rec_avg = np.mean(xview_rec_avg_seeds, axis=0)
        lgd = 'xview_RC{}'.format(rc_id)    
        ax_roc.plot(log_xtick, xview_rec_avg, linestyle='-.', marker=marker_list[ix], color=color_list[ix], linewidth=1, markersize=2.5, label=lgd)
    
        save_dir = save_path.format(xbs, 8-xbs)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        ax_roc.set_xticks(log_xtick)
        ax_roc.set_xticklabels(log_xtick)
        
        ax_roc.set_xlim(log_xtick[0]-0.05, log_xtick[-1]+0.05)
        ax_roc.set_title('ROC of RC {}'.format(eht), literal_eval(tlt_font))
        ax_roc.set_xlabel('log(FAR)', literal_eval(tlt_font))
        ax_roc.set_ylabel('Recall', literal_eval(tlt_font))
        #ax_roc.grid(True)
        
    handles, labels = ax_roc.get_legend_handles_labels()
    ax_roc.legend(handles, labels)
    #ax_roc.legend([A, B], legends)   
    #print('legends', legends)
    #plt.legend(legends, prop=literal_eval(lgd_font))
    #xview_yticks.append(1)
    #xview_yticks = list(dict.fromkeys(xview_yticks))
    #xview_yticks = list(set(xview_yticks))
    #print('yticks', xview_yticks)
    #plt.yticks(xview_yticks)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    save_name = 'syn_xview_only_RC1-5_ROC_AP{}_{}.png'.format(apN, eht)
    fig.savefig(os.path.join(save_dir, save_name), dpi=300)
    plt.close(fig)


def plot_syn_xview_cc_ROC_with_far_tick(ccids, syn_cmts, xbs_list, sxbs_list, syn_src_dir, xview_src_dir, syn_xview_src_dir, save_dir, xtick=[0.1, 1, 10]):
    eht = 'easy'
    syn_seeds = [0, 1, 2]
    xview_seeds = [0, 1]
    log_xtick = [np.log10(x) for x in xtick]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for ix, cc_id in enumerate(ccids):
        syn_rec_avg_seeds = []
        xview_rec_avg_seeds = []
        sx_rec_avg_seeds = []
        for sd in syn_seeds:
            syn_src_path = syn_src_dir.format(syn_cmts[ix], apN, sd)
            df_syn_rec = pd.read_csv(os.path.join(syn_src_path, 'rec_list.txt'), header=None)
            df_syn_far = pd.read_csv(os.path.join(syn_src_path, 'far_list.txt'), header=None)
            df_syn_far_0 = df_syn_far[df_syn_far<=xtick[0]]
            df_syn_far_0 = df_syn_far_0.dropna()
            syn_rec_0 = df_syn_rec.loc[df_syn_far_0.shape[0]-1]
            
            df_syn_far_1 = df_syn_far[df_syn_far<=xtick[1]]
            df_syn_far_1 = df_syn_far_1.dropna()
            syn_rec_1 = df_syn_rec.loc[df_syn_far_1.shape[0]-1]
            
            df_syn_far_2 = df_syn_far[df_syn_far<=xtick[2]]
            df_syn_far_2 = df_syn_far_2.dropna()
            syn_rec_2 = df_syn_rec.loc[df_syn_far_2.shape[0]-1]

            syn_rec_avg_seeds.append([syn_rec_0, syn_rec_1, syn_rec_2])
        syn_rec_avg_seeds = np.array(syn_rec_avg_seeds)
        syn_rec_avg = np.mean(syn_rec_avg_seeds, axis=0)
        
        for sd in xview_seeds:
            xview_src_path = xview_src_dir.format(xbs_list[ix], 8-xbs_list[ix], cc_id, apN, sd)
            df_xview_rec = pd.read_csv(os.path.join(xview_src_path, 'rec_list.txt'), header=None)
            df_xview_far = pd.read_csv(os.path.join(xview_src_path, 'far_list.txt'), header=None)
            df_xview_far_0 = df_xview_far[df_xview_far<=xtick[0]]
            df_xview_far_0 = df_xview_far_0.dropna()
            xview_rec_0 = df_xview_rec.loc[df_xview_far_0.shape[0]-1]
            
            df_xview_far_1 = df_xview_far[df_xview_far<=xtick[1]]
            df_xview_far_1 = df_xview_far_1.dropna()
            xview_rec_1 = df_xview_rec.loc[df_xview_far_1.shape[0]-1]
            
            df_xview_far_2 = df_xview_far[df_xview_far<=xtick[2]]
            df_xview_far_2 = df_xview_far_2.dropna()
            xview_rec_2 = df_xview_rec.loc[df_xview_far_2.shape[0]-1]
            
            xview_rec_avg_seeds.append([xview_rec_0, xview_rec_1, xview_rec_2])
        xview_rec_avg_seeds = np.array(xview_rec_avg_seeds)
        xview_rec_avg = np.mean(xview_rec_avg_seeds, axis=0)
        
        for sd in xview_seeds:
            syn_xview_src_path = syn_xview_src_dir.format(sxbs_list[ix], 8-sxbs_list[ix], cc_id, apN, sd)
            df_sx_rec = pd.read_csv(os.path.join(syn_xview_src_path, 'rec_list.txt'), header=None)
            df_sx_far = pd.read_csv(os.path.join(syn_xview_src_path, 'far_list.txt'), header=None)
            df_sx_far_0 = df_sx_far[df_sx_far<=xtick[0]]
            df_sx_far_0 = df_sx_far_0.dropna()
            sx_rec_0 = df_sx_rec.loc[df_sx_far_0.shape[0]-1]
            
            df_sx_far_1 = df_sx_far[df_sx_far<=xtick[1]]
            df_sx_far_1 = df_sx_far_1.dropna()
            sx_rec_1 = df_sx_rec.loc[df_sx_far_1.shape[0]-1]
            
            df_sx_far_2 = df_sx_far[df_sx_far<=xtick[2]]
            df_sx_far_2 = df_sx_far_2.dropna()
            sx_rec_2 = df_sx_rec.loc[df_sx_far_2.shape[0]-1]
            
            sx_rec_avg_seeds.append([sx_rec_0, sx_rec_1, sx_rec_2])
        sx_rec_avg_seeds = np.array(sx_rec_avg_seeds)
        sx_rec_avg = np.mean(sx_rec_avg_seeds, axis=0)
            
        fig, ax_roc = plt.subplots(1, 1)
        legends = ['syn_CC{}'.format(cc_id), 'xview_CC{}'.format(cc_id), 'syn+xview_CC{}'.format(cc_id)]
        
        ax_roc.plot(log_xtick, syn_rec_avg, linestyle='-.', marker=marker_list[0], color=color_list[0], linewidth=1, markersize=2.5)
        ax_roc.plot(log_xtick, xview_rec_avg, linestyle='-', marker=marker_list[1], color=color_list[1], linewidth=1, markersize=2.5)
        ax_roc.plot(log_xtick, sx_rec_avg, linestyle='--', marker=marker_list[2], color=color_list[2], linewidth=1, markersize=2.5)
        ax_roc.set_xticks(log_xtick)
        ax_roc.set_xticklabels(log_xtick)
        #lgd = 'xview_CC{}_AP{}_{}'.format(cc_id, apN, eht)
        ax_roc.set_xlim(log_xtick[0]-0.05, log_xtick[-1]+0.05)
        ax_roc.set_title('ROC of CC{} {}'.format(cc_id, eht), literal_eval(tlt_font))
        ax_roc.set_xlabel('log(FAR)', literal_eval(tlt_font))
        ax_roc.set_ylabel('Recall', literal_eval(tlt_font))
        #ax_roc.grid(True)
        #legends.append(lgd)

        fig.legend(legends, prop=literal_eval(lgd_font))
        #xview_yticks.append(1)
        #xview_yticks = list(dict.fromkeys(xview_yticks))
        #xview_yticks = list(set(xview_yticks))
        #print('yticks', xview_yticks)
        #plt.yticks(xview_yticks)
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
          
        save_name = 'syn_xview_CC{}_ROC_AP{}_{}.png'.format(cc_id, apN, eht)
        fig.savefig(os.path.join(save_dir, save_name), dpi=300)
        plt.close(fig)

if __name__ == '__main__':
    lgd_font = "{'family': 'serif', 'weight': 'normal', 'size': 8}"
    tlt_font = "{'family': 'serif', 'weight': 'normal', 'size': 13}"
    
    sd = 17
    # ap_list = [20, 40, 50]  # 20 #
    apN = 50
    ehtypes = ['easy']
    model_ids = [4, 1, 5, 5, 5]
    rare_classes = [1, 2, 3, 4, 5]
    
    #marker_list = ['-^', '-x', '-<', '->', '-o', '-*']
    ## baseline -- xview
    #save_dir = '../../result_output/1_cls/px23whr3_seed17/test_on_xview_hgiou1_1gpu_xview_only_rc_ap50/'  
    #plot_ROC_with_far_less_than_3(far_thres = 3)
    
    ######### syn RC VS. xview RC
#    marker_list = ['^', 'x', '*', '>', 'o']
#    color_list = ['r', 'g', 'royalblue', 'y', 'peru']
#    xbs_list = [7, 7, 7, 7, 6]
#    syn_cmts = ['syn_xview_bkg_px15whr3_xbw_newbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias0.03_RC1_v111_seed17',
#                'syn_xview_bkg_px23whr3_xbsw_xwing_newbkg_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_square_bias10_RC2_v116_seed17',
#                'syn_xview_bkg_px23whr3_xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.03_color_square_bias10_RC3_v116_seed17',
#                'syn_xview_bkg_px23whr3_xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias0.12_RC4_v114_seed17',
#                'syn_xview_bkg_px23whr3_xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias0.03_RC5_v111_seed17']
#    seed=0  
#    xview_src_dir = '/data/users/yang/code/yxu-yolov3-xview/result_output/1_cls/px23whr3_seed17/xview_only_oa/test_on_xview_ccnrc_bkg_aug_rc{}_hgiou1_19.5obj_x{}bg{}_iou{}_seed{}'
#    syn_src_dir = '/data/users/yang/code/yxu-yolov3-xview/result_output/1_cls/{}/test_on_xview_ccnrc_bkg_aug_rc{}_hgiou1_1gpu_val_syn_iou{}_seed{}'
#    save_dir = '/data/users/yang/code/yxu-yolov3-xview/result_output/1_cls/px23whr3_seed17/xview_only_oa/test_on_xview_ccnrc_bkg_aug_rc_hgiou1_19.5obj'
#    plot_syn_xview_rc_ROC_with_far_tick(xbs_list, xview_src_dir, syn_cmts, syn_src_dir, save_dir, xtick=[0.1, 1, 10])
    


    ########## zero-shot VS. real only VS. few-shot
    marker_list = ['^', 'x', '*']
    color_list = ['r', 'g', 'peru']
    ccids = [1, 2]
    syn_cmts = ['syn_xview_bkg_px23whr3_new_bkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.24_color_square_bias2_CC1_v47_seed17',
                'syn_xview_bkg_px23whr3_new_bkg_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias0.12_CC2_v42_seed17']             
    xbs_list = [7, 6] 
    sxbs_list = [6, 6]
    
    syn_src_dir = '/data/users/yang/code/yxu-yolov3-xview/result_output/1_cls/syn_CC/{}/test_on_xview_rcncc_bkg_cc_hgiou1_1gpu_val_syn_easy_iou{}_seed{}/'
    xview_src_dir = '/data/users/yang/code/yxu-yolov3-xview/result_output/1_cls/px23whr3_seed17/xview_CC/test_on_xview_rcncc_bkg_cc_hgiou1_19.5obj_cc{}x{}_ccid{}_easy_iou{}_seed{}/'
    syn_xview_src_dir = '/data/users/yang/code/yxu-yolov3-xview/result_output/1_cls/px23whr3_seed17/syn+xview_CC/test_on_xview_rcncc_bkg_cc_hgiou1_19.5obj_cc{}x{}_ccid{}_easy_iou{}_seed{}/'
    save_dir = '/data/users/yang/code/yxu-yolov3-xview/result_output/1_cls/px23whr3_seed17/syn+xview_CC/roc_curve_of_zero-real-fewshot/'

    plot_syn_xview_cc_ROC_with_far_tick(ccids, syn_cmts, xbs_list, sxbs_list, syn_src_dir, xview_src_dir, syn_xview_src_dir, save_dir, xtick=[0.1, 1, 10])
    
    