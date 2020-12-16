import os
import argparse
import glob
import shutil
import numpy as np
import pandas as pd
import sys
sys.path.append('/data/users/yang/code/yxu-yolov3-xview')
from utils.object_score_util import get_bbox_coords_from_annos_with_object_score as gbc



def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def split_bkg_into_train_val(comments='px{}whr{}_seed{}', seed=17):
    '''
    Other airplane(NCC, NRC) labels belong to BG
    split data contains no aircrafts (bkg images)
    '''
    comments = comments.format(px_thres, whr_thres, seed)
    bkg_lbl_dir = args.annos_save_dir[:-1] + '_bkg'
    bkg_img_dir = args.images_save_dir[:-1] + '_noairplane_bkg_chips'
    data_save_dir = args.data_save_dir
    if comments:
        txt_save_dir = args.data_list_save_dir + comments # + '_bh'+ '/'
        if not os.path.exists(txt_save_dir):
            os.makedirs(txt_save_dir)
        data_save_dir = os.path.join(data_save_dir, comments)
        if not os.path.exists(data_save_dir):
            os.makedirs(data_save_dir)
    else:
        txt_save_dir = args.data_list_save_dir

    ##### images that contain no aircrafts-- BKG
    bkg_lbl_files = glob.glob(os.path.join(bkg_lbl_dir, '*.txt'))
    print('bkg_lbl_files', len(bkg_lbl_files))
    bkg_lbl_files.sort()
    bkg_img_files = [os.path.join(bkg_img_dir, os.path.basename(f).replace('.txt', '.jpg')) for f in bkg_lbl_files]

    np.random.seed(seed)
    bkg_ixes = np.random.permutation(len(bkg_lbl_files))
    print('bkg_ixes[:10]', bkg_ixes[:10])
    val_bkg_num = int(len(bkg_lbl_files)*args.val_percent)
    trn_bkg_num = len(bkg_lbl_files) - val_bkg_num
    trn_bkg_lbl_files =[bkg_lbl_files[i] for i in bkg_ixes[:trn_bkg_num ]]
    val_bkg_lbl_files = [bkg_lbl_files[i] for i in bkg_ixes[trn_bkg_num: trn_bkg_num + val_bkg_num]]
    trn_bkg_img_files = [bkg_img_files[i] for i in bkg_ixes[: trn_bkg_num]]
    val_bkg_img_files = [bkg_img_files[i] for i in bkg_ixes[trn_bkg_num: trn_bkg_num + val_bkg_num]]
    print('trn_bkg_lbl_files', len(trn_bkg_lbl_files), len(trn_bkg_img_files))
    print('val_bkg_img_files', len(val_bkg_lbl_files), len(val_bkg_img_files))

    ###### train mixed batch of xview_rc + xview_nrc_bkg
    trn_bkg_img_txt = open(os.path.join(txt_save_dir, 'xview_bkg_train_img_{}.txt'.format(comments)), 'w')
    trn_bkg_lbl_txt = open(os.path.join(txt_save_dir, 'xview_bkg_train_lbl_{}.txt'.format(comments)), 'w')

    ###### validate xview_rc_nrc_bkg
    val_bkg_img_txt = open(os.path.join(txt_save_dir, 'xview_bkg_val_img_{}.txt'.format(comments)), 'w')
    val_bkg_lbl_txt = open(os.path.join(txt_save_dir, 'xview_bkg_val_lbl_{}.txt'.format(comments)), 'w')

    trn_bkg_lbl_dir = args.annos_save_dir[:-1] + '_trn_bkg_lbl'
    val_bkg_lbl_dir = args.annos_save_dir[:-1] + '_val_bkg_lbl'
    if os.path.exists(trn_bkg_lbl_dir):
        shutil.rmtree(trn_bkg_lbl_dir)
        os.mkdir(trn_bkg_lbl_dir)
    else:
        os.mkdir(trn_bkg_lbl_dir)
    if os.path.exists(val_bkg_lbl_dir):
        shutil.rmtree(val_bkg_lbl_dir)
        os.mkdir(val_bkg_lbl_dir)
    else:
        os.mkdir(val_bkg_lbl_dir)

    trn_bkg_img_dir = args.images_save_dir[:-1] + '_trn_bkg_img'
    val_bkg_img_dir = args.images_save_dir[:-1] + '_val_bkg_img'
    if os.path.exists(trn_bkg_img_dir):
        shutil.rmtree(trn_bkg_img_dir)
        os.mkdir(trn_bkg_img_dir)
    else:
        os.mkdir(trn_bkg_img_dir)

    if os.path.exists(val_bkg_img_dir):
        shutil.rmtree(val_bkg_img_dir)
        os.mkdir(val_bkg_img_dir)
    else:
        os.mkdir(val_bkg_img_dir)

    for i in range(len(trn_bkg_lbl_files)):
        trn_bkg_lbl_txt.write("%s\n" % trn_bkg_lbl_files[i])
        trn_bkg_img_txt.write("%s\n" % trn_bkg_img_files[i])
        lbl_name = os.path.basename(trn_bkg_lbl_files[i])
        img_name = lbl_name.replace('.txt', '.jpg')
        shutil.copy(os.path.join(bkg_lbl_dir, lbl_name), os.path.join(trn_bkg_lbl_dir, lbl_name))
        shutil.copy(os.path.join(bkg_img_dir, img_name), os.path.join(trn_bkg_img_dir, img_name))
    trn_bkg_lbl_txt.close()
    trn_bkg_img_txt.close()

    for j in range(len(val_bkg_lbl_files)):
        val_bkg_lbl_txt.write("%s\n" % val_bkg_lbl_files[j])
        val_bkg_img_txt.write("%s\n" % val_bkg_img_files[j])
        lbl_name = os.path.basename(val_bkg_lbl_files[j])
        img_name = lbl_name.replace('.txt', '.jpg')
        shutil.copy(os.path.join(bkg_lbl_dir, lbl_name), os.path.join(val_bkg_lbl_dir, lbl_name))
        shutil.copy(os.path.join(bkg_img_dir, img_name), os.path.join(val_bkg_img_dir, img_name))
    val_bkg_lbl_txt.close()
    val_bkg_img_txt.close()

    shutil.copy(os.path.join(txt_save_dir, 'xview_bkg_train_img_{}.txt'.format(comments)),
                    os.path.join(data_save_dir, 'xview_bkg_train_img_{}.txt'.format(comments)))
    shutil.copy(os.path.join(txt_save_dir, 'xview_bkg_val_img_{}.txt'.format(comments)),
                    os.path.join(data_save_dir, 'xview_bkg_val_img_{}.txt'.format(comments)))
    shutil.copy(os.path.join(txt_save_dir, 'xview_bkg_train_lbl_{}.txt'.format(comments)),
                    os.path.join(data_save_dir, 'xview_bkg_train_lbl_{}.txt'.format(comments)))
    shutil.copy(os.path.join(txt_save_dir, 'xview_bkg_val_lbl_{}.txt'.format(comments)),
                    os.path.join(data_save_dir, 'xview_bkg_val_lbl_{}.txt'.format(comments)))


def combine_xview_BG_with_syn_CC_by_bkgtimes(syn_cmt, times=2, seed=17):
    '''
    get Nx trn bkg images, 
    combine with synthetic data
    create Nx *.data
    '''
    syn_data_dir = args.syn_data_list_dir.format(syn_cmt, args.class_num)
    syn_img_file = os.path.join(syn_data_dir, '{}_seed{}'.format(syn_cmt, seed),  '{}_train_img_seed{}.txt'.format(syn_cmt, seed))
    syn_lbl_file = os.path.join(syn_data_dir, '{}_seed{}'.format(syn_cmt, seed), '{}_train_lbl_seed{}.txt'.format(syn_cmt, seed))
    df_syn_img = pd.read_csv(syn_img_file, header=None)
    syn_num = df_syn_img.shape[0]
    
    trn_bkg_lbl_dir = args.annos_save_dir[:-1] + '_trn_bkg_lbl'
    trn_bkg_img_dir = args.images_save_dir[:-1] + '_trn_bkg_img'
    bkg_imgs = np.sort(glob.glob(os.path.join(trn_bkg_img_dir, '*.jpg')))
    
    np.random.seed(seed)
    indexes = np.random.permutation(len(bkg_imgs))
    bkg_num = times*syn_num
    bkg_inxes = indexes[:bkg_num]
    
    base_cmt = 'px{}whr{}_seed{}'.format(px_thres, whr_thres, seed)
    data_save_dir =  os.path.join(args.data_save_dir, base_cmt)
    print('data_save_dir', data_save_dir)
    bkg_img_times = open(os.path.join(data_save_dir, 'trn_bkg_img_{}times_syn.txt'.format(times)), 'w')
    bkg_lbl_times = open(os.path.join(data_save_dir, 'trn_bkg_lbl_{}times_syn.txt'.format(times)), 'w')
    for ix in bkg_inxes:
        bkg_img_times.write('%s\n' % bkg_imgs[ix])
        lbl_name = os.path.basename(bkg_imgs[ix]).replace('.jpg', '.txt')
        bkg_lbl_times.write('%s\n' % os.path.join(trn_bkg_lbl_dir, lbl_name))
    bkg_img_times.close()
    bkg_lbl_times.close()
    
    cc_id = syn_cmt[syn_cmt.find('CC')+2]
    syn_save_dir =  os.path.join(data_save_dir, 'CC')
    data_txt = open(os.path.join(syn_save_dir, '{}_+xview_bkg_{}x_{}.data'.format(syn_cmt, times, base_cmt)), 'w')
    data_txt.write(
        'syn_train={}\n'.format(syn_img_file))
    data_txt.write(
        'syn_train_label={}\n'.format(syn_lbl_file))
        
    data_txt.write(
        'xview_bkg_train={}\n'.format(os.path.join(data_save_dir, 'trn_bkg_img_{}times_syn.txt'.format(times))))
    data_txt.write(
        'xview_bkg_train_label={}\n'.format(os.path.join(data_save_dir, 'trn_bkg_lbl_{}times_syn.txt'.format(times))))

    data_txt.write(
        'valid={}\n'.format(os.path.join(syn_data_dir, '{}_seed{}'.format(syn_cmt, seed),  '{}_val_img_seed{}.txt'.format(syn_cmt, seed))))
    data_txt.write(
        'valid_label={}\n'.format(os.path.join(syn_data_dir, '{}_seed{}'.format(syn_cmt, seed),  '{}_val_lbl_seed{}.txt'.format(syn_cmt, seed))))

    xview_trn_num = syn_num + bkg_num

    data_txt.write('xview_number={}\n'.format(xview_trn_num))
    data_txt.write('classes=%s\n' % str(args.class_num))
    data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(args.class_num))
    data_txt.write('backup=backup/\n')
    data_txt.write('eval=color\n')
    data_txt.close()


def combine_all_xview_BG_with_syn_CC_one_inst(syn_cmt, ins=6, sample_seeds=[0, 1], seed=17, val_syn=True):
    ''' 
    get all trn bkg images, 
    combine with synthetic data ins, contral the number of instances
    create  *.data
    '''
    syn_data_dir = args.syn_data_list_dir.format(syn_cmt, args.class_num)
    syn_img_file = os.path.join(syn_data_dir, '{}_seed{}'.format(syn_cmt, seed), '{}_train_img_seed{}.txt'.format(syn_cmt, seed))
    syn_lbl_file = os.path.join(syn_data_dir, '{}_seed{}'.format(syn_cmt, seed), '{}_train_lbl_seed{}.txt'.format(syn_cmt, seed))
    df_syn_img = pd.read_csv(syn_img_file, header=None)
    df_syn_lbl = pd.read_csv(syn_lbl_file, header=None)
    syn_num = df_syn_img.shape[0]

    base_cmt = 'px{}whr{}_seed{}'.format(px_thres, whr_thres, seed)
    data_save_dir =  os.path.join(args.data_save_dir, base_cmt)
    print('data_save_dir', data_save_dir)
    bkg_img = os.path.join(data_save_dir, 'xview_bkg_trn_img_{}.txt'.format(base_cmt))
    bkg_lbl = os.path.join(data_save_dir, 'xview_bkg_trn_lbl_{}.txt'.format(base_cmt))
    df_bkg_img = pd.read_csv(bkg_img, header=None)
    bkg_num = df_bkg_img.shape[0]
    bkg_img_val = os.path.join(data_save_dir, 'xview_bkg_val_img_{}.txt'.format(base_cmt))
    bkg_lbl_val = os.path.join(data_save_dir, 'xview_bkg_val_lbl_{}.txt'.format(base_cmt))
    df_bkg_img_val = pd.read_csv(bkg_img_val, header=None) 
    df_bkg_lbl_val = pd.read_csv(bkg_lbl_val, header=None)
    syn_img_val = os.path.join(syn_data_dir, '{}_seed{}'.format(syn_cmt, seed),  '{}_val_img_seed{}.txt'.format(syn_cmt, seed))
    syn_lbl_val = os.path.join(syn_data_dir, '{}_seed{}'.format(syn_cmt, seed),  '{}_val_lbl_seed{}.txt'.format(syn_cmt, seed))
    df_syn_img_val = pd.read_csv(syn_img_val, header=None)
    df_syn_lbl_val = pd.read_csv(syn_lbl_val, header=None)
    df_val_img = df_syn_img_val.append(df_bkg_img_val)
    df_val_lbl = df_syn_lbl_val.append(df_bkg_lbl_val)
    df_val_img.to_csv(os.path.join(syn_data_dir, '{}_seed{}'.format(syn_cmt, seed),  '{}+xview_bkg_val_img_seed{}.txt'.format(syn_cmt, seed)), header=False, index=False)
    df_val_lbl.to_csv(os.path.join(syn_data_dir, '{}_seed{}'.format(syn_cmt, seed),  '{}+xview_bkg_val_lbl_seed{}.txt'.format(syn_cmt, seed)), header=False, index=False)

    for ssd in sample_seeds:
        samp_cmt = 'px{}whr{}_seed{}'.format(px_thres, whr_thres, ssd)
        np.random.seed(ssd)
        inxes = np.random.permutation(syn_num)
        syn_inst_img_file = os.path.join(syn_data_dir, '{}_seed{}'.format(syn_cmt, seed), '{}_{}instance_train_img_seed{}.txt'.format(syn_cmt, ins, ssd))
        syn_inst_lbl_file = os.path.join(syn_data_dir, '{}_seed{}'.format(syn_cmt, seed), '{}_{}instance_train_lbl_seed{}.txt'.format(syn_cmt, ins, ssd))
        df_inst_img = df_syn_img.loc[inxes[:ins], 0]
        df_inst_lbl = df_syn_lbl.loc[inxes[:ins], 0]
        df_inst_img.to_csv(syn_inst_img_file, header=False, index=False)
        df_inst_lbl.to_csv(syn_inst_lbl_file, header=False, index=False)

        data_txt = open(os.path.join(syn_data_dir, '{}_seed{}'.format(syn_cmt, seed), '{}_{}instances+xview_bkg_seed{}.data'.format(syn_cmt, ins, ssd)), 'w')
        data_txt.write(
            'syn_train={}\n'.format(syn_inst_img_file))
        data_txt.write(
            'syn_train_label={}\n'.format(syn_inst_lbl_file))
            
        data_txt.write(
            'xview_bkg_train={}\n'.format(bkg_img))
        data_txt.write(
            'xview_bkg_train_label={}\n'.format(bkg_lbl))
    
        if val_syn:
            data_txt.write(
                'valid={}\n'.format(os.path.join(syn_data_dir, '{}_seed{}'.format(syn_cmt, seed),  '{}+xview_bkg_val_img_seed{}.txt'.format(syn_cmt, seed))))
            data_txt.write(
                'valid_label={}\n'.format(os.path.join(syn_data_dir, '{}_seed{}'.format(syn_cmt, seed),  '{}+xview_bkg_val_lbl_seed{}.txt'.format(syn_cmt, seed))))
        else:
            data_txt.write(
                'valid={}\n'.format(os.path.join(data_save_dir, 'xview_cc_rc_bkg_val_img_{}.txt'.format(base_cmt))))
            data_txt.write(
                'valid_label={}\n'.format(os.path.join(data_save_dir, 'xview_cc_rc_bkg_val_lbl_{}.txt'.format(base_cmt))))
    
        xview_trn_num = syn_num + bkg_num
    
        data_txt.write('xview_number={}\n'.format(xview_trn_num))
        data_txt.write('classes=%s\n' % str(args.class_num))
        data_txt.write('names=./data_xview/{}_cls/xview.names\n'.format(args.class_num))
        data_txt.write('backup=backup/\n')
        data_txt.write('eval=color\n')
        data_txt.close()


def create_syn_dataset_with_different_quantities():
    for ix, syn_cmt in enumerate(syn_cmts):
        syn_quan_dir = args.syn_cc_dir.format(cmt_quantities[ix])
        syn_annos_quan_dir = args.syn_annos_cc_dir.format(cmt_quantities[ix])
        print(syn_quan_dir)
        quan_img_files = np.sort(glob.glob(os.path.join(syn_quan_dir, 'color_all_images_step182.4', '*.png')))
        quan_lbl_files = np.sort(glob.glob(os.path.join(syn_annos_quan_dir, 'minr100_linkr15_px23whr3_color_all_annos_txt_step182.4','*.txt')))
        quan_img_num = len(quan_img_files) 
        #prefix = cmt[:cmt.find('_CC')+3]
        syn_data_dir = args.syn_data_list_dir.format(syn_cmt, args.class_num)
        syn_trn_img_file = os.path.join(syn_data_dir, '{}_seed{}'.format(syn_cmt, seed),  '{}_train_img_seed{}.txt'.format(syn_cmt, seed))
        syn_trn_lbl_file = os.path.join(syn_data_dir, '{}_seed{}'.format(syn_cmt, seed), '{}_train_lbl_seed{}.txt'.format(syn_cmt, seed))
        syn_val_img_file = os.path.join(syn_data_dir, '{}_seed{}'.format(syn_cmt, seed),  '{}_val_img_seed{}.txt'.format(syn_cmt, seed))
        syn_val_lbl_file = os.path.join(syn_data_dir, '{}_seed{}'.format(syn_cmt, seed), '{}_val_lbl_seed{}.txt'.format(syn_cmt, seed))
        df_src_trn_img = pd.read_csv(syn_trn_img_file, header=None)
        df_src_trn_lbl = pd.read_csv(syn_trn_lbl_file, header=None)
        src_num_trn = df_src_trn_lbl.shape[0] 
        df_src_val_img = pd.read_csv(syn_val_img_file, header=None)
        df_src_val_lbl = pd.read_csv(syn_val_lbl_file, header=None)
        src_num_val = df_src_val_img.shape[0] 
        for tx, t in enumerate([0.5, 2, 4]):
            quant = quantities[ix].format(base_version+tx, t)
            quan_data_file = '{}_seed{}.data'.format(quant, seed)
            quan_trn_img_file = '{}_train_img_seed{}.txt'.format(quant, seed)
            quan_trn_lbl_file = '{}_train_lbl_seed{}.txt'.format(quant, base_version+tx, seed)
            quan_val_img_file = '{}_val_img_seed{}.txt'.format(quant, base_version+tx, seed)
            quan_val_lbl_file = '{}_val_lbl_seed{}.txt'.format(quant, base_version+tx, seed)
            syn_quan_dir = args.syn_data_list_dir.format(quant, args.class_num)
            if not os.path.exists(syn_quan_dir):
                os.mkdir(syn_quan_dir)
            quan_data = open(os.path.join(syn_quan_dir, quan_data_file), 'w')
            new_trn_img_files = open(os.path.join(syn_quan_dir, quan_trn_img_file), 'w')
            new_trn_lbl_files = open(os.path.join(syn_quan_dir, quan_trn_lbl_file), 'w')
            new_val_img_files = open(os.path.join(syn_quan_dir, quan_val_img_file), 'w')
            new_val_lbl_files = open(os.path.join(syn_quan_dir, quan_val_lbl_file), 'w')
            new_num_trn = int(src_num_trn*t)
            new_num_val = int(src_num_val*t)       
            
            if t < 1:
                print('t<1 new_num_trn', new_num_trn)
                for i in range(new_num_trn):
                    new_trn_img_files.write('%s\n'%(df_src_trn_img.loc[i, 0]))
                    new_trn_lbl_files.write('%s\n'%(df_src_trn_lbl.loc[i, 0]))
                for j in range(new_num_val):
                    new_val_img_files.write('%s\n'%(df_src_val_img.loc[j, 0]))
                    new_val_lbl_files.write('%s\n'%(df_src_val_lbl.loc[j, 0]))
                    
            else:
                print('new_num_trn', new_num_trn)
                for i in range(src_num_trn):
                    new_trn_img_files.write('%s\n'%(df_src_trn_img.loc[i, 0]))
                    new_trn_lbl_files.write('%s\n'%(df_src_trn_lbl.loc[i, 0]))
                for j in range(src_num_val):
                    new_val_img_files.write('%s\n'%(df_src_val_img.loc[j, 0]))
                    new_val_lbl_files.write('%s\n'%(df_src_val_lbl.loc[j, 0]))
                    
                np.random.seed(seed)
                med_num_trn = new_num_trn-src_num_trn
                med_num_val = new_num_val-src_num_val
                print('med_num_trn', med_num_trn)
                indexes = np.random.permutation(quan_img_num)
                print('indexes', len(indexes))
                trn_ixes = indexes[:med_num_trn]
                val_ixes = indexes[med_num_trn:med_num_trn+med_num_val]
                for i in trn_ixes:
                    
                    new_trn_img_files.write('%s\n'%(quan_img_files[i]))
                    new_trn_lbl_files.write('%s\n'%(quan_lbl_files[i]))
                for j in val_ixes:
                    new_val_img_files.write('%s\n'%(quan_img_files[j]))
                    new_val_lbl_files.write('%s\n'%(quan_lbl_files[j]))
                
            new_trn_img_files.close()
            new_trn_lbl_files.close()
            new_val_img_files.close()
            new_val_img_files.close()
                
            quan_data.write('train={}\n'.format( os.path.join(syn_quan_dir, quan_trn_img_file)))
            quan_data.write('train_label={}\n'.format(os.path.join(syn_quan_dir, quan_trn_lbl_file)))
            quan_data.write('valid={}\n'.format( os.path.join(syn_quan_dir, quan_val_img_file)))
            quan_data.write('valid_label={}\n'.format(os.path.join(syn_quan_dir, quan_val_lbl_file)))
            quan_data.write('xview_number={}\n'.format(new_num_trn))
            quan_data.write('classes=%s\n' % str(args.class_num))
            quan_data.write('names=./data_xview/{}_cls/xview.names\n'.format(args.class_num))
            quan_data.write('backup=backup/\n')
            quan_data.write('eval=quantities')
            quan_data.close()

 

def get_args(px_thres=None, whr_thres=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--xview_yolo_dir", type=str, help="dir to xViewYOLO",
                        default='/data/users/yang/data/xView_YOLO/')

    parser.add_argument("--images_save_dir", type=str, help="to save chip trn val images files",
                        default='/data/users/yang/data/xView_YOLO/images/')

    # parser.add_argument("--images_save_dir", type=str, help="to save chip trn val images files",
    #                     default='/data/users/yang/data/xView_YOLO/images/{}_')

    parser.add_argument("--syn_save_dir", type=str, help="",
                        default='/data/users/yang/data/synthetic_data/')
                        
    parser.add_argument("--syn_data_list_dir", type=str, help="to syn data list files",
                        default='/data/users/yang/code/yxu-yolov3-xview/data_xview/{}_{}_cls/')
                        
    parser.add_argument("--syn_cc_dir", type=str,
                        help="Path to folder containing synthetic images and annos ",
                        default='/data/users/yang/data/synthetic_data_CC/{}/')
    parser.add_argument("--syn_annos_cc_dir", type=str, default='/data/users/yang/data/synthetic_data_CC/{}_txt_xcycwh/',
                        help="syn xview txt")

    parser.add_argument("--txt_save_dir", type=str, help="to save  related label files",
                        default='/data/users/yang/data/xView_YOLO/labels/')

    parser.add_argument("--annos_save_dir", type=str, help="to save txt annotation files",
                        default='/data/users/yang/data/xView_YOLO/labels/')

    parser.add_argument("--data_save_dir", type=str, help="to save data files",
                        default='/data/users/yang/code/yxu-yolov3-xview/data_xview/{}_cls/')
    parser.add_argument("--data_list_save_dir", type=str, help="to save data files",
                        default='/data/users/yang/data/xView_YOLO/labels/{}/{}_cls/data_list/')
    parser.add_argument("--cat_sample_dir", type=str, help="to save figures",
                        default='/data/users/yang/data/xView_YOLO/cat_samples/')
    parser.add_argument("--val_percent", type=float, default=0.2,
                        help="0.24 0.2 Percent to split into validation (ie .25 = val set is 25% total)")
    parser.add_argument("--font3", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 10}")
    parser.add_argument("-ft2", "--font2", type=str, help="legend font",
                        default="{'family': 'serif', 'weight': 'normal', 'size': 13}")
    parser.add_argument("--class_num", type=int, default=1, help="Number of Total Categories")  # 60  6 1
    parser.add_argument("--seed", type=int, default=17, help="random seed") #fixme ---- 1024 17
    parser.add_argument("--input_size", type=int, default=608, help="Number of Total Categories")  # 300 416

    args = parser.parse_args()
    #fixme ----------==--------change
    args.images_save_dir = args.images_save_dir + '{}_{}cls/'.format(args.input_size, args.class_num)
    args.annos_save_dir = args.annos_save_dir + '{}/{}_cls_xcycwh_px{}whr{}/'.format(args.input_size, args.class_num, px_thres, whr_thres)
    args.txt_save_dir = args.txt_save_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)
    args.data_save_dir = args.data_save_dir.format(args.class_num)
    args.data_list_save_dir = args.data_list_save_dir.format(args.input_size, args.class_num)
    if not os.path.exists(args.txt_save_dir):
        os.makedirs(args.txt_save_dir)

    if not os.path.exists(args.annos_save_dir):
        os.makedirs(args.annos_save_dir)
    if not os.path.exists(args.data_list_save_dir):
        os.makedirs(args.data_list_save_dir)
    if not os.path.exists(args.images_save_dir):
        os.makedirs(args.images_save_dir)

    if not os.path.exists(args.data_save_dir):
        os.makedirs(args.data_save_dir)
    if not os.path.exists(args.data_list_save_dir):
        os.makedirs(args.data_list_save_dir)
    args.cat_sample_dir = args.cat_sample_dir + '{}/{}_cls/'.format(args.input_size, args.class_num)
    if not os.path.exists(args.cat_sample_dir):
        os.makedirs(args.cat_sample_dir)
    return args


if __name__ == '__main__':
    px_thres=23
    whr_thres=3
    args = get_args(px_thres, whr_thres)

    '''
    split BKG images into train and val
    '''
#    comments = 'px{}whr{}_seed{}'
#    px_thres = 23
#    whr_thres = 3
#    seed = 17
#    split_bkg_into_train_val(comments, seed)


    '''
    combine xivew_BG with syn CC
    '''
#    seed=17
#    base_cmt='px{}whr{}_seed{}'.format(px_thres, whr_thres, seed)
#    syn_cmts = ['syn_xview_bkg_px23whr3_new_bkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.24_color_square_bias2_CC1_v47',
#    'syn_xview_bkg_px23whr3_new_bkg_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias0.12_CC2_v42']
#    times = [1, 2, 4]
#    for ti in times:
#        for syn_cmt in syn_cmts:
#            combine_xview_BG_with_syn_CC_by_bkgtimes(syn_cmt, ti, seed)

    seed=17
    base_cmt='px{}whr{}_seed{}'.format(px_thres, whr_thres, seed)
#    syn_cmts = ['syn_xview_bkg_px23whr3_new_bkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0_color_square_bias0_CC1_1inst_v63',
#    'syn_xview_bkg_px23whr3_new_bkg_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias0_CC2_1inst_v63']
#    syn_cmts = ['syn_xview_bkg_px23whr3_new_bkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.24_color_square_bias2_CC1_1inst_v64',
#    'syn_xview_bkg_px23whr3_new_bkg_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias0.12_CC2_1inst_v64']
#    instances = [6,12,18,24,30]
#    sample_seeds = [0, 1]
#    for ti in instances:
#        for syn_cmt in syn_cmts:
#            #combine_all_xview_BG_with_syn_CC_one_inst(syn_cmt, ti, seed)
#            combine_all_xview_BG_with_syn_CC_one_inst(syn_cmt, ti, sample_seeds, seed, val_syn=False)

    syn_cc1_cmt = 'syn_xview_bkg_px23whr3_new_bkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.08_csig1_CC1_1inst_v65'
    cc1_instances = [7, 14, 21, 28, 35]#[6,12,18,24,30]
    instances = cc1_instances
    syn_cmt = syn_cc1_cmt
#    syn_cc2_cmt = 'syn_xview_bkg_px23whr3_new_bkg_shdw_split_scatter_gauss_rndsolar_promu_ssig0_csig4_CC2_1inst_v65'
#    cc2_instances = [7, 14, 21, 28, 35]
#    instances = cc2_instances
#    syn_cmt = syn_cc2_cmt
    sample_seeds = [2] # 0, 1
    for ti in instances:
        #combine_all_xview_BG_with_syn_CC_one_inst(syn_cmt, ti, seed)
        combine_all_xview_BG_with_syn_CC_one_inst(syn_cmt, ti, sample_seeds, seed, val_syn=False)
 
    '''
     quantities
     train: 180  360  720  1440
     val: 45  90  180  360
    '''
#    syn_cmts = ['syn_xview_bkg_px23whr3_new_bkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.24_color_square_bias2_CC1_v47',
#    'syn_xview_bkg_px23whr3_new_bkg_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias0.12_CC2_v42']
#    
#    quantities = ['syn_xview_bkg_px23whr3_new_bkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.24_color_square_bias2_CC1_v{}_{}quantities',
#    'syn_xview_bkg_px23whr3_new_bkg_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias0.12_CC2_v{}_{}quantities']
#    
#    cmt_quantities = ['syn_xview_bkg_new_bkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.24_color_square_bias2_CC1_quantities',
#    'syn_xview_bkg_new_bkg_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias0.12_CC2_quantities']
#    
#    seed = 17
#    base_version = 60
#    create_syn_dataset_with_different_quantities()

    
    
    
