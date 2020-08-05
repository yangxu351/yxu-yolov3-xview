import argparse
import json
import numpy as np
from torch.utils.data import DataLoader

from utils.parse_config_xview import *
from models_xview import *

from utils.datasets_xview import *
# from utils.datasets_xview_backup import *
from utils.utils_xview import *


# import sys
# sys.path.append('.')


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


#fixme
# def get_val_imgid_by_name(image_name, opt=None):
#     image_id_name_maps = json.load(
#         open(os.path.join(opt.label_dir, 'all_image_ids_names_dict_{}cls.json'.format(opt.class_num))))
#     img_ids = [int(k) for k in image_id_name_maps.keys()]
#     img_names = [v for v in image_id_name_maps.values()]
#     return img_ids[img_names.index(image_name)]


def get_val_imgid_by_name(path, name):
    json_img_id_file = glob.glob(os.path.join(path, 'xview_val*_img_id_map.json'))[0]
    img_id_map = json.load(open(json_img_id_file))
    imgIds = [id for id in img_id_map.values()]
    return img_id_map[name]
    # # print(path)
    # val_files = pd.read_csv(path, header=None).to_numpy()
    # # print('val_files 0', val_files[0])
    # val_names = [os.path.basename(vf[0]) for vf in val_files]
    # img_id = val_names.index(name)
    # return img_id


def test(cfg,
         data,
         weights=None,
         batch_size=16,
         img_size=416,
         conf_thres=0.001,
         nms_iou_thres=0.5,  # for nms
         save_json=False,
         model=None,
         dataloader=None,
         opt=None):
    if model is None:
        device = torch_utils.select_device(opt.device, batch_size=batch_size)
        verbose = opt.task == 'test'

        # Remove previous
        for f in glob.glob('test_batch*.jpg'):
            os.remove(f)

        # Initialize model
        model = Darknet(cfg, img_size).to(device)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
#            m_key = 215# opt.epochs -1
#            model.load_state_dict(torch.load(weights, map_location=device)[m_key]['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:  # called by train.py
        device = next(model.parameters()).device  # get model device
        verbose = False

    # Configure run
    data = parse_data_cfg(data)
    nc = int(data['classes'])  # number of classes
    # fixme
#    path = data['valid']  # path to test images
#    lbl_path = data['valid_label']
    path = data['test']  # path to test images
    lbl_path = data['test_label']
    # path = data['valid_rare']  # path to test images
    # lbl_path = data['valid_rare_label']
    names = load_classes(data['names'])  # class names
    if apN == 20:
        iouv = torch.linspace(0.2, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    else:
        iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
       
    iouv = iouv[0].view(1)  # for mAP@0.5
    niou = iouv.numel()

        # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels(path, lbl_path, img_size, batch_size, rect=True, cache_labels=True, with_modelid=False)  #
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    # fixme
    # coco91class = coco80_to_coco91_class()
    xview_classes = np.arange(nc)
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3)
    jdict, stats, ap, ap_class = [], [], [], []
    # fixme
    tcls = []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        # print('targets--', targets.shape)
        # print('paths--', len(paths), paths)
        # # print('shapes', shapes)
        # print('targets', targets)
        # exit(0)
        _, _, height, width = imgs.shape  # batch size, channels, height, width

        # Plot images with bounding boxes
        if batch_i == 0 and not os.path.exists('test_1gpu.jpg'):
            plot_images(imgs=imgs, targets=targets, paths=paths, fname='test_batch0.jpg')

        # Disable gradients
        with torch.no_grad():
            # Run model
            inf_out, train_out = model(imgs)  # inference and training outputs

            # Compute loss
            if hasattr(model, 'hyp'):  # if model has loss hyperparameters
                loss += compute_loss(train_out, targets, model)[1][:3].cpu()  # GIoU, obj, cls
            # print('inf_out', inf_out)
            # Run NMS
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=nms_iou_thres)
            # print('output', output)
        # Statistics per image
        # print('opt.model_id', opt.model_id)
        for si, pred in enumerate(output):
            # print('si', si, targets[si])
            labels = targets[targets[:, 0] == si, 1:]
            # print('labels', labels.shape)
            # print('labels', labels)
            #fixme --yang.xu
            # if opt.model_id is not None:
            #     nl = len(labels)
            #     if nl:
            #         labels = labels[labels[:, -1] == opt.model_id]
            #         nl = len(labels)
            #     tcls =labels[:, -1].tolist() if nl else []
            # else:
            #     nl = len(labels)
            #     tcls = labels[:, 0].tolist() if nl else []  # target class
            #fixme --yang.xu
            # labels = targets[targets[:, 0] == si, 1:]
            # print('labels', labels)
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            #fixme --yang.xu
            # tcls = labels[:, 0].tolist() if nl else []  # target class
            # print('tcls', tcls)

            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, 1), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                # fixme
                # print('paths[si]', paths[si]) #  /media/lab/Yang/data/xView_YOLO/images/608/1094_11.jpg
                # image_id = int(Path(paths[si]).stem.split('_')[-1])

                image_name = paths[si].split('/')[-1]
                # image_id = get_val_imgid_by_name(opt.base_dir, image_name)

                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner # xtlytlwh
                for di, d in enumerate(pred):
                    #fixme
                    # jdict.append({'image_id': image_id,
                    #               'category_id': xview_classes[int(d[5])],
                    #               'bbox': [floatn(x, 3) for x in box[di]],
                    #               'score': floatn(d[4], 5)})  # conf
                    jdict.append({'image_name': image_name, # image_id,
                                  'category_id': xview_classes[int(d[5])],
                                  'bbox': [floatn(x, 3) for x in box[di]],
                                  'score': floatn(d[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(len(pred), niou, dtype=torch.bool)
            if nl:
                detected = []  # target indices

                tcls_tensor = labels[:, 0]
                #fixme --yang.xu
                # if opt.model_id is not None:
                #     tcls_tensor = labels[:, -1]
                #     print(tcls_tensor)
                # else:
                #     tcls_tensor = labels[:, 0]
                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * torch.Tensor([width, height, width, height]).to(device)

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices
                    # print('ti', ti, 'pi ', pi)

                    # Search for detections
                    if len(pi):
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            # pred (x1, y1, x2, y2, object_conf, conf, class)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
            # print('\n correct: {}  pred[:,4]:{}  pred[:, 5]:{} tcls:{}'.format(correct, pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats, pr_path=opt.result_dir, pr_name=opt.name)
        # if niou > 1:
        #       p, r, ap, f1 = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # average across ious
        #fixme --yang.xu
        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Save JSON
    if save_json and map and len(jdict):
        # fixme
        # img_names = [os.path.basename(x) for x in dataloader.dataset.img_files]
        #fixme
        # img_id_maps = json.load(
        #     open(os.path.join(opt.label_dir, 'all_image_ids_names_dict_{}cls.json'.format(opt.class_num))))
        # img_id_list = [k for k in img_id_maps.keys()]
        # img_name_list = [v for v in img_id_maps.values()]
        # imgIds = [img_id_list[img_name_list.index(v)] for v in img_name_list if
        #           v in img_names]  # note: index is the same as the keys
        # sids = set(imgIds)
        # print('imgIds', len(imgIds), 'sids', len(sids))

        # json_img_id_file = glob.glob(os.path.join(opt.base_dir, 'xview_val*_img_id_map.json'))[0]
        # img_id_map = json.load(open(json_img_id_file))
        # imgIds = [id for id in img_id_map.values()]

        # imgIds = [get_val_imgid_by_name(na) for na in img_names]
        # sids = set(imgIds)
        # print('imgIds', len(imgIds), 'sids', len(sids))
        # imgIds = np.arange(len(output))

        result_json_file = 'results_{}.json'.format(opt.name)
        with open(os.path.join(opt.result_dir, result_json_file), 'w') as file:
            # json.dump(jdict, file)
            json.dump(jdict, file, ensure_ascii=False, indent=2, cls=MyEncoder)


    # Return results
    maps = np.zeros(nc) + map
    # print('ap', ap, 'ap_class', ap_class)
    #fixme
    # for i, c in enumerate(ap_class):
    #     maps[c] = ap[i]
    for i in range(len(ap_class)):
        maps[i] = ap[i]
    return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps
    # return (mp, mr, map, mf1, *(loss / len(dataloader)).tolist()), maps
    # return (mp, mr, map, mf1, *(loss / len(dataloader)).tolist())

def get_opt(dt=None, sr=None, comments=''):
    parser = argparse.ArgumentParser() # prog='test.py'

    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-{}cls_syn.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data_xview/{}_cls/{}/xview_{}_{}.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='weights/{}_cls/{}_{}/best_{}_{}.pt', help='path to weights file')

    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch') # 2
    parser.add_argument('--img_size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--res', type=float, default=0.3, help='resolution')
    parser.add_argument('--margin', type=int, default=30, help='margin size (pixels)')

    parser.add_argument('--class_num', type=int, default=1, help='class number')  # 60 6
    parser.add_argument('--label_dir', type=str, default='/media/lab/Yang/data/xView_YOLO/labels/', help='*.json path')
    parser.add_argument('--weights_dir', type=str, default='weights/{}_cls/{}_seed{}/', help='to save weights path')
    parser.add_argument('--result_dir', type=str, default='result_output/{}_cls/{}_seed{}/{}/', help='to save result files path')
    parser.add_argument('--grids_dir', type=str, default='grids_dir/{}_cls/{}_seed{}/', help='to save grids images')
    parser.add_argument("--syn_ratio", type=float, default=sr, help="ratio of synthetic data: 0 0.25, 0.5, 0.75")
    parser.add_argument('--syn_display_type', type=str, default=dt, help='syn_texture0, syn_color0, syn_texture, syn_color, syn_mixed, syn')
    parser.add_argument('--base_dir', type=str, default='data_xview/{}_cls/{}/', help='without syn data path')

    parser.add_argument('--conf-thres', type=float, default=0.01, help='0.1 0.05 0.001 object confidence threshold')
    parser.add_argument('--nms-iou-thres', type=float, default=0.5, help='NMS 0.5  0.6 IOU threshold for NMS')
    parser.add_argument('--save_json', action='store_true', default=True, help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--device', default='1', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--name', default='', help='name')
    parser.add_argument('--cmt', default=comments, help='comments')
    parser.add_argument('--model_id', type=int, default=None, help='specified model id')
    parser.add_argument('--rare_class', type=int, default=None, help='specified rare class')
    parser.add_argument('--type', default='hard', help='hard, easy')
    opt = parser.parse_args()
    # opt.save_json = opt.save_json or any([x in opt.data for x in ['xview.data']])
    opt.cfg = opt.cfg.format(opt.class_num)

    return opt


if __name__ == '__main__':

    
#    comments = ['syn_uspp_bkg_shdw_split_scatter_gauss_rotate_wnd_v2_color']
#    comments = ['syn_uspp_bkg_shdw_split_scatter_gauss_rotate_rnd_ort_wnd_v3_color']
    comments = ['syn_uspp_ca_az_bkg_shdw_split_scatter_gauss_rotate_rnd_ort_wnd_v4_color']
    base_cmt = 'px10_seed{}'
    test_cmt = 'wnd_CA_AZ'
    # hyp_cmt = 'hgiou1_1gpu'

    # hyp_cmt = 'hgiou1_1gpu_obj29.5'
    # hyp_cmt = 'hgiou1_1gpu_xval'
    # hyp_cmt = 'hgiou1_mean_best'
    # hyp_cmt = 'hgiou1_obj3.5_val_labeled'
    # hyp_cmt = 'hgiou1_1gpu_val_labeled_miss'
    # hyp_cmt = 'hgiou1_1gpu_val_labeled'
#    hyp_cmt = 'hgiou1_1gpu_trans_val_syn'
#    prefix = 'syn_trans'


#    hyp_cmt = 'hgiou1_lr0.001_val_syn'
#    prefix = 'syn_lr0.001'
    
#    bkup = '100'
#    bkup = '150'
    bkup = '200'
    apN = 20
#    apN = 50

#    hyp_cmt = 'hgiou1_1gpu_val_syn'
#    prefix = 'syn_wnd_backup{}_ap{}'.format(bkup, apN)
#    prefix = 'syn_wnd'
    hyp_cmt = 'hgiou1_half_affine_val_syn'
    prefix = 'syn_wnd_hafl_affine_backup{}_ap{}'.format(bkup, apN)
    px_thres = 10
    sd = 17
    eh_types = ['hard']#, 'easy']
    for typ in eh_types:
        for cmt in comments:
            base_cmt = base_cmt.format(sd)
            opt = get_opt(comments=cmt)
            opt.device = '0'

            cinx = cmt.find('model') # first letter index
            endstr = cmt[cinx:]
            rcinx = endstr.rfind('_')
            fstr = endstr[rcinx:] # '_' is included
            sstr = endstr[:rcinx]
            suffix = fstr + '_' + sstr
            if cinx < 0:
                suffix = ''
            opt.name = prefix + suffix

            ''' for specified model id '''
            opt.batch_size = 8
            sidx = cmt.split('model')[-1][0]

            opt.conf_thres = 0.01
            tif_name = 'wnd_backup{}_ap{}'.format(bkup, apN)
            ############# 2 images test set
#            opt.type = 'easy'
#            opt.type = 'hard'
#            opt.type = typ
#            opt.rare_class = 1
#            opt.rare_class = 2
#            opt.rare_class = 3
#            opt.rare_class = 4 
#            opt.rare_class = 5
            opt.rare_class = 6
            
            opt.name = tif_name
            opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_wnd_{}_{}'.format(hyp_cmt, tif_name))

#            opt.data = 'data_wnd/{}/wndtest_{}.data'.format(base_cmt, base_cmt)
            opt.data = 'data_wnd/{}/{}_{}.data'.format(base_cmt, test_cmt, base_cmt)

            ''' for whole validation dataset '''
            # opt.conf_thres = 0.1
            # opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd , '{}_{}_seed{}'.format('test_on_xview_with_model', hyp_cmt, sd))
            # opt.data = 'data_xview/{}_cls/{}/{}_seed{}_with_model.data'.format(opt.class_num, 'px{}whr{}_seed{}'.format(px_thres, whr_thres, sd), 'xview_px{}whr{}'.format(px_thres, whr_thres), sd)

            if not os.path.exists(opt.result_dir):
                os.makedirs(opt.result_dir)
            print(os.path.join(opt.weights_dir.format(opt.class_num, cmt, sd), '*_{}_seed{}'.format(hyp_cmt, sd), 'best_seed{}.pt'.format(sd)))
#            print(glob.glob(os.path.join(opt.weights_dir.format(opt.class_num, cmt, sd), '*_{}_seed{}'.format(hyp_cmt, sd), 'best_seed{}.pt'.format(sd))))

#            all_weights = glob.glob(os.path.join(opt.weights_dir.format(opt.class_num, cmt, sd), '*_{}_seed{}'.format(hyp_cmt, sd), 'best_*seed{}.pt'.format(sd)))
            all_weights = glob.glob(os.path.join(opt.weights_dir.format(opt.class_num, cmt, sd), '*_{}_seed{}'.format(hyp_cmt, sd), 'backup{}.pt'.format(bkup)))
            all_weights.sort()
            opt.weights = all_weights[-1]

            print(opt.weights)
            print(opt.data)
            test(opt.cfg,
                 opt.data,
                 opt.weights,
                 opt.batch_size,
                 opt.img_size,
                 opt.conf_thres,
                 opt.nms_iou_thres,
                 opt.save_json, opt=opt)


