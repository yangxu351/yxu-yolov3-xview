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
    device = torch_utils.select_device(opt.device, batch_size=batch_size)
    # model_maps = torch.load(weights, map_location=device)
    # last_num = 5
    # mp_arr = np.zeros((last_num))
    # mr_arr = np.zeros((last_num))
    # map_arr = np.zeros((last_num))
    # mf1_arr = np.zeros((last_num))
    # Configure run
    data = parse_data_cfg(data)
    nc = int(data['classes'])  # number of classes
    # fixme--yang.xu
    path = data['test']  # path to test images
    lbl_path = data['test_label']
    names = load_classes(data['names'])  # class names
#    m_key = 215
#    for ix, mk in enumerate([m_key]):
#     for ix, mk in enumerate(model_maps.keys()):
    # Initialize/load model and set device
    if model is None:
        device = torch_utils.select_device(opt.device, batch_size=batch_size)
        verbose = opt.task == 'test'

        # Remove previous
        for f in glob.glob('test_batch*.jpg'):
            os.remove(f)

        # Initialize model
        model = Darknet(cfg, img_size).to(device)

        # Load weights
        # attempt_download(weights)
        # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
        # else:  # darknet format
        #     _ = load_darknet_weights(model, model_maps[mk])

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:  # called by train.py
        device = next(model.parameters()).device  # get model device
        verbose = False

    if apN == 20:
        iouv = torch.linspace(0.2, 0.95, 10).to(device)  # iou vector for mAP@0.2:0.95
    else:
        iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    #fixme --yang.xu
    iouv = iouv[0].view(1)  # for mAP@0.5
    # iouv = iouv[3].view(1)  # for mAP@0.65
    # iouv = iouv[5].view(1)  # for mAP@0.75
    niou = iouv.numel()

    # Dataloader
    if dataloader is None:
        #fixme --Yang.xu
        # dataset = LoadImagesAndLabels(path, lbl_path, img_size, batch_size, rect=True, cache_labels=True)  #
        if opt.model_id is not None:
            dataset = LoadImagesAndLabels(path, lbl_path, img_size, batch_size, rect=True, cache_labels=True, with_modelid=True)
        else:
            dataset = LoadImagesAndLabels(path, lbl_path, img_size, batch_size, rect=True, cache_labels=True)
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
    if apN == 20:
        s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.2', 'F1')
    else:
        s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3)
    jdict, stats, ap, ap_class = [], [], [], []
    # fixme --yang.xu
    sum_labels = 0
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)

        # print('imgs--', imgs.shape)
        # print('targets--', targets.shape)
        # print('targets ', targets)
        # print('paths--', len(paths))
        # print('shapes', len(shapes))
        # exit(0)
        _, _, height, width = imgs.shape  # batch size, channels, height, width

        # Plot images with bounding boxes
        if batch_i == 0 and not os.path.exists('test_1gpu.jpg'):
            plot_images(imgs=imgs, targets=targets, paths=paths, fname='test_batch0.jpg')

        # Disable gradients
        with torch.no_grad():
            # Run model
            inf_out, train_out = model(imgs)  # inference and training outputs
            # plot_grids(train_out, batch_i, paths=paths, save_dir=opt.grids_dir)
            # exit(0)

            # Compute loss
            if hasattr(model, 'hyp'):  # if model has loss hyperparameters
                loss += compute_loss(train_out, targets, model)[1][:3].cpu()  # GIoU, obj, cls

            # Run NMS
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=nms_iou_thres)
        # Statistics per image
        for si, pred in enumerate(output):
            # print('si ', si, targets[si])
            # print('targets--', targets.shape)
            labels = targets[targets[:, 0] == si, 1:]
            # # print('labels--', labels.shape)
            # # print(labels)
            nl = len(labels)
            sum_labels += nl
            #fixme ---yang.xu
            # tcls = labels[:, 0].tolist() if nl else []  # target class
            tcls = labels[:, -1].tolist() if nl else []
            # print('tcls ', len(tcls))
            # exit(0)

            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, 1), torch.Tensor(), torch.Tensor(), tcls, torch.zeros(0, 1)))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))
            #fixme --yang.xu
            pred = drop_boundary(pred, (height, width), margin_thres=opt.margin)
            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                #fixme
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
            neu_correct = torch.zeros(len(pred), niou, dtype=torch.bool)
            if nl:
                detected = []  # target indices

                #fixme --yang.xu
                tcls_tensor = labels[:, -1]
                # print('tcls_tensor', tcls_tensor)
                # exit(0)

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * torch.Tensor([width, height, width, height]).to(device)

                # Per target class
                # for cls in torch.unique(tcls_tensor):

                #fixme --yang.xu
                # ti = (cls == tcls_tensor).nonzero().view(-1)  # target indices
                # pi = (cls == pred[:, 5]).nonzero().view(-1)  # prediction indices
                #fixme --yang.xu
                ti = (opt.rare_class == tcls_tensor).nonzero().view(-1) # target indices
                pi = (0 == pred[:, 5]).nonzero().view(-1)  # prediction indices
#                    if len(ti):
#                        print('\nti ', len(ti), ti)

                if opt.type == 'easy':
                    neu_cls = 0
                    ni = (neu_cls == tcls_tensor).nonzero().view(-1) # target neutral indices
#                        print('ni ', len(ni), ni)
                else:
                    ni = torch.tensor([])

                # if len(pi):
                #     ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices
                #     # Append detections
                #     for j in (ious > iouv[0]).nonzero():
                #         d = ti[i[j]]  # detected target
                #         if d not in detected:
                #             detected.append(d)
                #             correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                #             if len(detected) == nl:  # all targets already located in image
                #                 break

                # Search for detections
                if len(pi) and len(ti) and not len(ni):
                    ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices
                    # print('ious ', ious.shape)
                    # Append detections
                    for j in (ious > iouv[0]).nonzero():
                        d = ti[i[j]]  # detected target
                        # print('d', d)
                        if d not in detected:
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break
                elif len(pi) and not len(ti) and len(ni):
                    neu_ious, nix = box_iou(pred[pi, :4], tbox[ni]).max(1)
                    for s in (neu_ious > iouv[0]).nonzero():
#                            print('s --------> ', s)
                        d = ni[nix[s]]  # detected target
                        if d not in detected:
                            detected.append(d)
                            neu_correct[pi[s]] = neu_ious[s] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break
                elif len(pi) and len(ti) and len(ni):
                    ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices
                    neu_ious, nix = box_iou(pred[pi, :4], tbox[ni]).max(1)
                    for j in (ious > iouv[0]).nonzero():
                        d = ti[i[j]]  # detected target
                        if d not in detected:
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break
                    for s in (neu_ious > iouv[0]).nonzero():
#                            print('s --------> ', s)
                        d = ni[nix[s]]  # detected neutral target
                        if d not in detected:
                            detected.append(d)
                            neu_correct[pi[s]] = neu_ious[s] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break
#                    print('detected ', detected)
            # Append statistics (correct, conf, pcls, tcls, neu_correct)
            # pred (x1, y1, x2, y2, object_conf, conf, class)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 5].cpu(), tcls, neu_correct))
#                if len(tcls) and len(correct):
#                    print('\n correct: {}  pred[:,4]:{}  pred[:, 5]:{} tcls:{}'.format(correct, pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
            # if len(tcls) and len(neu_correct):
#                    print('\n neu_correct: {}  pred[:,4]:{}  pred[:, 5]:{} tcls:{}'.format(neu_correct, pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    print('sum all labels', sum_labels)
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    if len(stats):
        pr_name= opt.name + '  @IoU: {:.2f} '.format(iouv[0]) + '  conf_thres: {} '.format(conf_thres)
        # print('*stats', *stats)
        p, r, ap, f1, ap_class = ap_per_class(*stats, pr_path=opt.result_dir, pr_name= pr_name, rare_class=opt.rare_class)

        print('dataset.batch ', dataset.batch.shape)
        # exit(0)
        area = (img_size*opt.res)*(img_size*opt.res)*dataset.batch.shape[0]*1e-6

        plot_roc_easy_hard(*stats, pr_path=opt.result_dir, pr_name= pr_name, rare_class=opt.rare_class, area=area, ehtype=opt.type, title_data_name=tif_name)
        # if niou > 1:
        #       p, r, ap, f1 = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # average across ious
        #fixme --yang.xu
        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        #fixme --yang.xu compute before
        # nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        st3 = stats[3][stats[3] == opt.rare_class]
        nt = np.bincount(st3.astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)


    #fixme--yang.xu
    # Print results
    # pf = '%20s' + '%10.3g' * 6  # print format
    # print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    pf = '%20s' + '%10.3g' * 6  # print format
    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Save JSON
    if save_json and map and len(jdict):
        result_json_file = 'results_{}_bkup{}.json'.format(opt.name, bkup)
        with open(os.path.join(opt.result_dir, result_json_file), 'w') as file:
            # json.dump(jdict, file)
            json.dump(jdict, file, ensure_ascii=False, indent=2, cls=MyEncoder)
    
    bkup_file.write('%d %d %d %1.6f %1.6f %1.6f %1.6f\n' % (bkup, seen, nt.sum(), mp, mr, map, mf1))
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))
     # Return results
    maps = np.zeros(nc) + map
    # print('ap', ap, 'ap_class', ap_class)
    #fixme
    for i in range(len(ap_class)):
        maps[i] = ap[i]
    return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps


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

    comments = ["syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_scatter_gauss_40_bias0_model5_v1_color", "syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_scatter_gauss_40_color_bias76.5_model5_v4_color", "syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_scatter_gauss_40_color_bias178.5_model5_v8_color", "syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_scatter_gauss_40_color_bias255.0_model5_v11_color"]
    model_id = 5
    rare_cls = 3
    #
    # comments = ["syn_xview_bkg_px23whr3_xbsw_xwing_scatter_gauss_30_bias0_model1_v1_color", "syn_xview_bkg_px23whr3_xbsw_xwing_scatter_gauss_30_color_bias76.5_model1_v4_color", "syn_xview_bkg_px23whr3_xbsw_xwing_scatter_gauss_30_color_bias178.5_model1_v8_color", "syn_xview_bkg_px23whr3_xbsw_xwing_scatter_gauss_30_color_bias255.0_model1_v11_color"]
#    model_id = 1
#    rare_cls = 2

#    comments = ["syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_rnd_color_bias0_model4_v21_color", "syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_rnd_color_bias127.5_model4_v26_color", "syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_rnd_color_bias178.5_model4_v28_color", "syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_rnd_color_bias229.5_model4_v30_color"]
#    model_id = 4
#    rare_cls = 1

    base_cmt = 'px23whr3_seed{}'

#    hyp_cmt = 'hgiou1_lr0.001_val_syn'
#    prefix = 'syn_lr0.001'
    apN = 20
    hyp_cmt = 'hgiou1_1gpu_val_syn'

    px_thres = 23
    whr_thres = 3 # 4
    sd = 17
    eh_types = ['hard', 'easy']
    for typ in eh_types:
        for cmt in comments:
            bkup_file_name = '{}_iou{}_bkups_{}.txt'.format(cmt, apN, typ)
            bkup_file = open(os.path.join('result_output/1_cls/{}_seed{}/'.format(cmt, sd), bkup_file_name), 'w')
            for bkup in range(5, 220, 5):
                prefix = 'syn_iou{}_bkup{}'.format(apN, bkup)
                base_cmt = base_cmt.format(sd)
                opt = get_opt(comments=cmt)
                opt.device = '2'

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
                opt.model_id = int(sidx)
                print('opt.model_id', opt.model_id)
                opt.conf_thres = 0.01
                tif_name = 'xview'
                opt.type = typ
                opt.rare_class = rare_cls

    #            opt.name += '_{}'.format(opt.type)
    #            opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_xview_{}_upscale_m{}_rc{}_{}'.format(hyp_cmt, opt.model_id, opt.rare_class, opt.type))
    #            opt.data = 'data_xview/{}_cls/{}/xviewtest_{}_upscale_m{}_rc{}_{}.data'.format(opt.class_num, base_cmt, base_cmt, opt.model_id, opt.rare_class, opt.type)

                opt.name += '_{}_bkup{}'.format(opt.type, bkup)
    #            opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_xview_{}_m{}_rc{}_{}'.format(hyp_cmt, opt.model_id, opt.rare_class, opt.type))
                opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_xview_{}_m{}_rc{}_{}_iou{}_bkup{}'.format(hyp_cmt, opt.model_id, opt.rare_class, opt.type, apN, bkup))
                opt.data = 'data_xview/{}_cls/{}/xviewtest_{}_m{}_rc{}_{}.data'.format(opt.class_num, base_cmt, base_cmt, opt.model_id, opt.rare_class, opt.type)

    #            opt.name += '_{}_aug'.format(opt.type)
    #            opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_xview_{}_m{}_rc{}_{}_aug'.format(hyp_cmt, opt.model_id, opt.rare_class, opt.type))
    #            opt.data = 'data_xview/{}_cls/{}/xviewtest_{}_m{}_rc{}_{}_aug.data'.format(opt.class_num, base_cmt, base_cmt, opt.model_id, opt.rare_class, opt.type)

    #            opt.name += '_{}'.format(opt.type)
    #            opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_xview_{}_m{}_rc{}_{}'.format(hyp_cmt, opt.model_id, opt.rare_class, opt.type))
    #            opt.data = 'data_xview/{}_cls/{}/xviewtest_{}_m{}_rc{}_{}.data'.format(opt.class_num, base_cmt, base_cmt, opt.model_id, opt.rare_class, opt.type)

    #            opt.name += '_{}_aug'.format(opt.type)
    #            opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_xview_{}_m{}_rc{}_{}_aug'.format(hyp_cmt, opt.model_id, opt.rare_class, opt.type))
    #            opt.data = 'data_xview/{}_cls/{}/xviewtest_{}_m{}_rc{}_{}_aug.data'.format(opt.class_num, base_cmt, base_cmt, opt.model_id, opt.rare_class, opt.type)


    #            opt.grids_dir = opt.grids_dir.format(opt.class_num, cmt, sd)
    #            if not os.path.exists(opt.grids_dir):
    #                os.makedirs(opt.grids_dir)
                ############ all 2315 patches 171
    #            tif_name = '2315'
    #            opt.batch_size = 8
    #            opt.rare_class = 1
    #            opt.type = 'hard'
    #            opt.name += '_{}_{}'.format(tif_name, opt.type)
    #            opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_xview_with_model_{}_{}_{}'.format(hyp_cmt, tif_name, opt.type))
    #            opt.data = 'data_xview/{}_cls/{}/xviewtest_{}_m{}_rc{}_{}.data'.format(opt.class_num, base_cmt, base_cmt, opt.model_id, opt.rare_class, tif_name)

                ''' for whole validation dataset '''
                # opt.conf_thres = 0.1
                # opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd , '{}_{}_seed{}'.format('test_on_xview_with_model', hyp_cmt, sd))
                # opt.data = 'data_xview/{}_cls/{}/{}_seed{}_with_model.data'.format(opt.class_num, 'px{}whr{}_seed{}'.format(px_thres, whr_thres, sd), 'xview_px{}whr{}'.format(px_thres, whr_thres), sd)

                if not os.path.exists(opt.result_dir):
                    os.makedirs(opt.result_dir)
    #            print(os.path.join(opt.weights_dir.format(opt.class_num, cmt, sd), '*_{}_seed{}'.format(hyp_cmt, sd), 'best_seed{}.pt'.format(sd)))
    #            print(glob.glob(os.path.join(opt.weights_dir.format(opt.class_num, cmt, sd), '*_{}_seed{}'.format(hyp_cmt, sd), 'best_seed{}.pt'.format(sd))))
#                all_weights = glob.glob(os.path.join(opt.weights_dir.format(opt.class_num, cmt, sd), '*_{}_seed{}'.format(hyp_cmt, sd), 'best_*seed{}.pt'.format(sd)))
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

            bkup_file.close()
