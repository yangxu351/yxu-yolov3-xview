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
    model_maps = torch.load(weights, map_location=device)
    last_num = 5 
    mp_arr = np.zeros((last_num))
    mr_arr = np.zeros((last_num))
    map_arr = np.zeros((last_num))
    mf1_arr = np.zeros((last_num))
    # Configure run
    data = parse_data_cfg(data)
    nc = int(data['classes'])  # number of classes
    # fixme--yang.xu
    path = data['test']  # path to test images
    lbl_path = data['test_label']
    names = load_classes(data['names'])  # class names
#    m_key = 215
#    for ix, mk in enumerate([m_key]):
    for ix, mk in enumerate(model_maps.keys()):
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
            model.load_state_dict(model_maps[mk]['model'])
            # else:  # darknet format
            #     _ = load_darknet_weights(model, model_maps[mk])

            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
        else:  # called by train.py
            device = next(model.parameters()).device  # get model device
            verbose = False


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
        mp_arr[ix] = mp
        mr_arr[ix] = mr
        map_arr[ix] = map
        mf1_arr[ix] = mf1

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

            result_json_file = 'results_{}_{}.json'.format(opt.name, mk)
            with open(os.path.join(opt.result_dir, result_json_file), 'w') as file:
                # json.dump(jdict, file)
                json.dump(jdict, file, ensure_ascii=False, indent=2, cls=MyEncoder)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp_arr.mean(), mr_arr.mean(), map_arr.mean(), mf1_arr.mean()))

        # try:
        #     from pycocotools.coco import COCO
        #     from pycocotools.cocoeval import COCOeval
        # except:
        #     print('WARNING: missing pycocotools package, can not compute official COCO mAP. See requirements.txt.')
        #
        #     # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        #     # fixme
        #     # cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
        #     cocoGt = COCO(glob.glob(os.path.join(opt.base_dir, '*_xtlytlwh.json'))[0])
        #     cocoDt = cocoGt.loadRes(os.path.join(opt.result_dir, result_json_file))  # initialize COCO pred api
        #
        #     cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        #     cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
        #     cocoEval.evaluate()
        #     cocoEval.accumulate()
        #     cocoEval.summarize()
        #     mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)

    # Return results
    # maps = np.zeros(nc) + map
    # for i, c in enumerate(ap_class):
    #     maps[c] = ap[i]
    # return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps
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

    '''
    test for syn_xveiw_background_*  on original
    '''
    # import train_syn_xview_background_seeds_loop_1cls as tsl
    # # comments = ['syn_xview_background_texture', 'syn_xview_background_color', 'syn_xview_background_mixed']
    # comments = ['syn_xview_bkg_certain_models_texture', 'syn_xview_bkg_certain_models_color', 'syn_xview_bkg_certain_models_mixed']
    # # comments = ['syn_xview_bkg_px20whr4_certain_models_texture', 'syn_xview_bkg_px20whr4_certain_models_color', 'syn_xview_bkg_px20whr4_certain_models_mixed']
    # seeds = [17]
    # #
    # #5  all        94       219    0.0699     0.466     0.175     0.122
    # for sd in seeds[:1]:
    #     for cmt in comments[:2]:
    #         opt = tsl.get_opt(sd, cmt, Train=False)
    #         opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd , '{}_hgiou1_seed{}'.format('test_on_original', sd))
    #         if not os.path.exists(opt.result_dir):
    #             os.mkdir(opt.result_dir)
    #
    #         opt.weights = glob.glob(os.path.join(opt.weights_dir.format(opt.class_num, cmt, sd, '*_hgiou1_seed{}'.format(sd)), 'best_{}_seed{}.pt'.format(cmt, sd)))[-1]
    #         opt.data = 'data_xview/{}_cls/{}/{}_seed{}.data'.format(opt.class_num, 'px6whr4_ng0_seed{}'.format(sd), 'xview_px6whr4_ng0', sd)
    #         # opt.data = 'data_xview/{}_cls/{}/{}_seed{}.data'.format(opt.class_num, 'px20whr4_seed{}'.format(sd), 'xview_px20whr4', sd)
    #         print(opt.data)
    #         opt.name = 'seed{}_on_original'.format(opt.seed)
    #         opt.task = 'test'
    #         opt.save_json = True
    #         test(opt.cfg,
    #              opt.data,
    #              opt.weights,
    #              opt.batch_size,
    #              opt.img_size,
    #              opt.conf_thres,
    #              opt.nms_iou_thres,
    #              opt.save_json, opt=opt)

    '''
    test for syn_xveiw_background_*  on original 
    '''
    # import train_syn_xview_background_seeds_loop_1cls as tsl
    # # comments = ['syn_xview_background_texture', 'syn_xview_background_color', 'syn_xview_background_mixed']
    # # comments = ['syn_xview_bkg_certain_models_texture', 'syn_xview_bkg_certain_models_color', 'syn_xview_bkg_certain_models_mixed']
    # # comments = ['syn_xview_bkg_px20whr4_certain_models_texture', 'syn_xview_bkg_px20whr4_certain_models_color', 'syn_xview_bkg_px20whr4_certain_models_mixed']
    # comments = ['syn_xview_bkg_px23whr4_scale_models_texture', 'syn_xview_bkg_px23whr4_scale_models_color', 'syn_xview_bkg_px23whr4_scale_models_mixed']
    # seeds = [17]
    # #
    # #5  all        94       219    0.0699     0.466     0.175     0.122
    # for sd in seeds[:1]:
    #     for cmt in comments[:1]:
    #         opt = tsl.get_opt(sd, cmt, Train=False)
    #         opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd , '{}_hgiou1_seed{}'.format('test_on_original', sd))
    #         if not os.path.exists(opt.result_dir):
    #             os.mkdir(opt.result_dir)
    #
    #         opt.weights = glob.glob(os.path.join(opt.weights_dir.format(opt.class_num, cmt, sd, '*_hgiou1_seed{}'.format(sd)), 'best_{}_seed{}.pt'.format(cmt, sd)))[0]
    #         # opt.data = 'data_xview/{}_cls/{}/{}_seed{}.data'.format(opt.class_num, 'px6whr4_ng0_seed{}'.format(sd), 'xview_px6whr4_ng0', sd)
    #         # opt.data = 'data_xview/{}_cls/{}/{}_seed{}.data'.format(opt.class_num, 'px20whr4_seed{}'.format(sd), 'xview_px20whr4', sd)
    #         opt.data = 'data_xview/{}_cls/{}/{}_seed{}.data'.format(opt.class_num, 'px23whr4_seed{}'.format(sd), 'xview_px23whr4', sd)
    #         print(opt.data)
    #         opt.name = 'seed{}_on_xview_with_model'.format(opt.seed)
    #         opt.task = 'test'
    #         opt.save_json = True
    #         test(opt.cfg,
    #              opt.data,
    #              opt.weights,
    #              opt.batch_size,
    #              opt.img_size,
    #              opt.conf_thres,
    #              opt.nms_iou_thres,
    #              opt.save_json, opt=opt)


    '''
    test for syn_xveiw_background_*_with_model
    '''
    # # comments = ['syn_xview_background_texture', 'syn_xview_background_color', 'syn_xview_background_mixed']
    # # comments = ['syn_xview_bkg_certain_models_texture', 'syn_xview_bkg_certain_models_color', 'syn_xview_bkg_certain_models_mixed']
    # # comments = ['syn_xview_bkg_px20whr4_certain_models_texture', 'syn_xview_bkg_px20whr4_certain_models_color', 'syn_xview_bkg_px20whr4_certain_models_mixed']
    # comments = ['syn_xview_bkg_px23whr4_scale_models_texture', 'syn_xview_bkg_px23whr4_scale_models_color', 'syn_xview_bkg_px23whr4_scale_models_mixed']
    # comments = ['syn_xview_bkg_px23whr4_small_models_color']
    # comments = ['syn_xview_bkg_px23whr4_small_fw_models_color']
    # comments = ['syn_xview_bkg_px23whr3_6groups_models_color', 'syn_xview_bkg_px23whr3_6groups_models_mixed']
    # comments = ['syn_xview_bkg_px23whr3_rnd_bwratio_models_color', 'syn_xview_bkg_px23whr3_rnd_bwratio_models_mixed']
    # # hyp_cmt = 'hgiou1_fitness'
    # comments = ['syn_xview_bkg_px23whr3_6groups_models_color', 'syn_xview_bkg_px23whr3_6groups_models_mixed']
    # comments = ['syn_xview_bkg_px23whr3_rnd_bwratio_models_color', 'syn_xview_bkg_px23whr3_rnd_bwratio_models_mixed']
    # comments = ['syn_xview_bkg_px23whr3_rnd_bwratio_flat0.8_models_color', 'syn_xview_bkg_px23whr3_rnd_bwratio_flat0.8_models_mixed']
    # comments = ['syn_xview_bkg_px23whr3_rnd_bwratio_asx_models_color', 'syn_xview_bkg_px23whr3_rnd_bwratio_asx_models_mixed']
    # comments = ['syn_xview_bkg_px23whr3_xratio_xcolor_models_color', 'syn_xview_bkg_px23whr3_xratio_xcolor_models_mixed']
    # comments = ['syn_xview_bkg_px23whr3_sbwratio_xratio_xcolor_models_color', 'syn_xview_bkg_px23whr3_sbwratio_xratio_xcolor_models_mixed']
    # comments = ['syn_xview_bkg_px23whr3_sbwratio_new_xratio_xcolor_models_color', 'syn_xview_bkg_px23whr3_sbwratio_new_xratio_xcolor_models_mixed']
    # comments = ['syn_xview_bkg_px23whr3_sbw_xcolor_model0_color', 'syn_xview_bkg_px23whr3_sbw_xcolor_model0_mixed']
    # comments = ['syn_xview_bkg_px15whr3_sbw_xcolor_model4_color', 'syn_xview_bkg_px15whr3_sbw_xcolor_model4_mixed']
    # comments = ['syn_xview_bkg_px15whr3_sbw_xcolor_model4_v1_color', 'syn_xview_bkg_px15whr3_sbw_xcolor_model4_v1_mixed']
    # comments = ['syn_xview_bkg_px15whr3_sbw_xcolor_model4_v2_color', 'syn_xview_bkg_px15whr3_sbw_xcolor_model4_v2_mixed']
    # comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_gauss_model4_v3_color', 'syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_gauss_model4_v3_mixed']
    # comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_gauss_model4_v4_color']
    # comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_gauss_model4_v4_mixed']
    # comments = ['syn_xview_bkg_px15whr3_sbw_xcolor_model4_v2_mixed']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_model4_v7_color','syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_model4_v7_mixed']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_rndp_model4_v7_color','syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_rndp_model4_v7_mixed']
#    comments = ['syn_xview_bkg_px15whr3_xbw_rndcolor_xbkg_unif_mig21_model4_v8_color','syn_xview_bkg_px15whr3_xbw_rndcolor_xbkg_unif_mig21_model4_v8_mixed']
#    comments = ['syn_xview_bkg_px15whr3_xbw_rndcolor_xbkg_unif_mig21_rndp_model4_v9_color','syn_xview_bkg_px15whr3_xbw_rndcolor_xbkg_unif_mig21_rndp_model4_v9_mixed']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_rndp_shdw_model4_v10_color','syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_rndp_shdw_model4_v10_mixed']
#    comments = ['syn_xview_bkg_px15whr3_xbw_rndcolor_xbkg_unif_mig21_rndp_shdw_model4_v11_color','syn_xview_bkg_px15whr3_xbw_rndcolor_xbkg_unif_mig21_rndp_shdw_model4_v11_mixed']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_rndp_shdw_dense_model4_v12_color','syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_rndp_shdw_dense_model4_v12_mixed']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_rndp_shdw_dense_angle_model4_v13_color','syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_rndp_shdw_dense_angle_model4_v13_mixed']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_rndp_shdw_dense_angle_fwc_model4_v14_color','syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_rndp_shdw_dense_angle_fwc_model4_v14_mixed']
#    comments = ['syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_mig21_rndp_shdw_dense_angle_fwc_model4_v14_color','syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_mig21_rndp_shdw_dense_angle_fwc_model4_v14_mixed']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_angle_fwc_scatter_gauss_dense_model4_v15_color','syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_angle_fwc_scatter_gauss_dense_model4_v15_mixed']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_angle_fwc_scatter_gauss_dense_model4_v16_color','syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_angle_fwc_scatter_gauss_dense_model4_v16_mixed']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_gauss_dense_model4_v17_color','syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_gauss_dense_model4_v17_mixed']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_gauss_dense_solar_model4_v18_color','syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_gauss_dense_solar_model4_v18_mixed']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_uniform_50_model4_v19_color','syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_uniform_50_model4_v19_mixed']
#    comments = ['syn_xview_bkg_px50whr3_xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_uniform_50_model4_v19_color','syn_xview_bkg_px50whr3_xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_uniform_50_model4_v19_mixed']
#    comments = ['syn_xview_bkg_px50whr3_xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_gauss_120_model4_v21_color','syn_xview_bkg_px50whr3_xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_gauss_120_model4_v21_mixed']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_gauss_120_model4_v21_color','syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_gauss_120_model4_v21_mixed']
#    comments = ['syn_xview_bkg_px30whr3_xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_gauss_120_model4_v21_upscale_color','syn_xview_bkg_px30whr3_xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_gauss_120_model4_v21_upscale_mixed']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_uniform_50_rdegree_model4_v22_color','syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_fwc_scatter_uniform_50_rdegree_model4_v22_mixed']
#    comments = ['syn_xview_bkg_px15whr3_xbw_rndcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_model4_v23_color','syn_xview_bkg_px15whr3_xbw_rndcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_model4_v23_mixed']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_bias0_model4_v1_color','syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_bias0_model4_v1_mixed']
#    comments = ['syn_xview_bkg_px30whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_bias0_model4_v1_color','syn_xview_bkg_px30whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_bias0_model4_v1_mixed' 
    comments = ['syn_xview_bkg_px30whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_bias0_model4_v1_color']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_100_bias0_model4_v2_color','syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_100_bias0_model4_v2_mixed']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias20_model4_v3_color']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias40_model4_v4_color']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias50_model4_v9_color']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias60_model4_v8_color']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias70_model4_v5_color']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias110_model4_v6_color']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_rnd_model4_v7_color']

#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_bias0_model4_v1_color']

#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias_all_model4_v20_color']    
#    comments = ['syn_xview_bkg_px30whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_bias0_model4_v1_color','syn_xview_bkg_px30whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_bias0_model4_v1_mixed'
    comments = ['syn_xview_bkg_px30whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_bias0_model4_v1_color']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_100_bias0_model4_v2_color']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias20_model4_v3_color']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias40_model4_v4_color']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias50_model4_v5_color']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias60_model4_v6_color']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias70_model4_v7_color']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias110_model4_v8_color']
#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_rnd_model4_v9_color']

#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_bias0_model4_v1_color']

#    comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_mig21_shdw_scatter_uniform_50_angle_bias_all_model4_v20_color']
    model_id = 4
    # comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_gauss_model4_v4_color', 'syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_gauss_model4_v4_mixed']
    # comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_unif_model4_v6_color']
    # comments = ['syn_xview_bkg_px23whr3_xbw_xrxc_spr_sml_gauss_models_color', 'syn_xview_bkg_px23whr3_xbw_xrxc_spr_sml_gauss_models_mixed']
    # comments = ['syn_xview_bkg_px23whr3_sbw_xcolor_model1_color', 'syn_xview_bkg_px23whr3_sbw_xcolor_model1_mixed']
    # comments = ['syn_xview_bkg_px23whr3_xbw_xcolor_gauss_model1_v1_color', 'syn_xview_bkg_px23whr3_xbw_xcolor_gauss_model1_v1_mixed']
    # comments = ['syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_gauss_model1_v2_color', 'syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_gauss_model1_v2_mixed']
    # comments = ['syn_xview_bkg_px23whr3_sbw_xcolor_xbkg_unif_model1_v3_color', 'syn_xview_bkg_px23whr3_sbw_xcolor_xbkg_unif_model1_v3_mixed']
    # comments = ['syn_xview_bkg_px23whr3_xbsw_xcolor_xbkg_gauss_model1_v4_color', 'syn_xview_bkg_px23whr3_xbsw_xcolor_xbkg_gauss_model1_v4_mixed']
#    comments = ['syn_xview_bkg_px23whr3_xbsw_xcolor_xbkg_unif_rndp_model1_v6_color', 'syn_xview_bkg_px23whr3_xbsw_xcolor_xbkg_unif_rndp_model1_v6_mixed']
#    comments = ['syn_xview_bkg_px23whr3_xbsw_xwing_color_xbkg_unif_rndp_model1_v8_color', 'syn_xview_bkg_px23whr3_xbsw_xwing_color_xbkg_unif_rndp_model1_v8_mixed']
#    comments = ['syn_xview_bkg_px23whr3_xbsw_xwing_color_xbkg_unif_rndp_shdw_model1_v9_color', 'syn_xview_bkg_px23whr3_xbsw_xwing_color_xbkg_unif_rndp_shdw_model1_v9_mixed']
#    model_id = 1
#    comments = ['syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_scatter_uniform_model5_v1_color','syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_scatter_uniform_model5_v1_mixed']
#    comments = ['syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_rndp_model5_v2_color','syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_rndp_model5_v2_mixed']
#    model_id = 5
    base_cmt = 'px23whr3_seed{}'
    # hyp_cmt = 'hgiou1_1gpu'

    # hyp_cmt = 'hgiou1_1gpu_obj29.5'
    # hyp_cmt = 'hgiou1_1gpu_xval'
    # hyp_cmt = 'hgiou1_mean_best'
    # hyp_cmt = 'hgiou1_obj3.5_val_labeled'
    # hyp_cmt = 'hgiou1_1gpu_val_labeled_miss'
    # hyp_cmt = 'hgiou1_1gpu_val_labeled'
#    hyp_cmt = 'hgiou1_1gpu_trans_val_syn'
#    prefix = 'syn_trans'

    hyp_cmt = 'hgiou1_1gpu_val_syn'
    prefix = 'syn'
#    prefix = 'syn_px30'

#    hyp_cmt = 'hgiou1_1gpu_anchor_val_syn'
#    prefix = 'syn_anchor'

#    hyp_cmt = 'hyp_tuned_val_syn'
#    prefix = 'syn_tuned'

#    hyp_cmt = 'hgiou1_1gpu_noaffine_val_syn'
#    prefix = 'syn_noaffine'

#    hyp_cmt = 'hgiou1_half_affine_val_syn'
#    prefix = 'syn_half_affine'

#    hyp_cmt = 'hgiou1_1gpu_noaug_val_syn'
#    prefix = 'syn_noaug'

#    hyp_cmt = 'hgiou1_obj39.5_val_syn'
#    prefix = 'syn_obj39.5'
    # hyp_cmt = 'hgiou1_1gpu_val_xview'
    # hyp_cmt = 'hgiou1_obj15.5_val_xview'
    # hyp_cmt = 'hgiou0.7_1gpu'

    px_thres = 23
    whr_thres = 3 # 4
    sd = 17
    eh_types = ['hard', 'easy']
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
            opt.model_id = int(sidx)
            print('opt.model_id', opt.model_id)
            opt.conf_thres = 0.01
            tif_name = 'xview'
            ############# 2 images test set
#            opt.type = 'easy'
#            opt.type = 'hard'
            opt.type = typ
            opt.rare_class = 1
#            opt.rare_class = 2
#            opt.rare_class = 3
#            opt.name += '_{}'.format(opt.type)
#            opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_xview_{}_upscale_m{}_rc{}_{}'.format(hyp_cmt, opt.model_id, opt.rare_class, opt.type))
#            opt.data = 'data_xview/{}_cls/{}/xviewtest_{}_upscale_m{}_rc{}_{}.data'.format(opt.class_num, base_cmt, base_cmt, opt.model_id, opt.rare_class, opt.type)

#            opt.name += '_{}'.format(opt.type)
#            opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_xview_{}_m{}_rc{}_{}'.format(hyp_cmt, opt.model_id, opt.rare_class, opt.type))
#            opt.data = 'data_xview/{}_cls/{}/xviewtest_{}_m{}_rc{}_{}.data'.format(opt.class_num, base_cmt, base_cmt, opt.model_id, opt.rare_class, opt.type)

#            opt.name += '_{}_aug'.format(opt.type)
#            opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_xview_{}_m{}_rc{}_{}_aug'.format(hyp_cmt, opt.model_id, opt.rare_class, opt.type))
#            opt.data = 'data_xview/{}_cls/{}/xviewtest_{}_m{}_rc{}_{}_aug.data'.format(opt.class_num, base_cmt, base_cmt, opt.model_id, opt.rare_class, opt.type)

            opt.name += '_{}'.format(opt.type)
            opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_xview_{}_m{}_rc{}_{}'.format(hyp_cmt, opt.model_id, opt.rare_class, opt.type))
            opt.data = 'data_xview/{}_cls/{}/xviewtest_{}_m{}_rc{}_{}.data'.format(opt.class_num, base_cmt, base_cmt, opt.model_id, opt.rare_class, opt.type)

            # opt.name += '_{}_aug'.format(opt.type)
            # opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_xview_{}_m{}_rc{}_{}_aug'.format(hyp_cmt, opt.model_id, opt.rare_class, opt.type))
            # opt.data = 'data_xview/{}_cls/{}/xviewtest_{}_m{}_rc{}_{}_aug.data'.format(opt.class_num, base_cmt, base_cmt, opt.model_id, opt.rare_class, opt.type)


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
            all_weights = glob.glob(os.path.join(opt.weights_dir.format(opt.class_num, cmt, sd), '*_{}_seed{}'.format(hyp_cmt, sd), 'best_*seed{}.pt'.format(sd)))
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


    '''
    test for xview_syn_xview_bkg_* with model
    '''
    # # # # comments = ['xview_syn_xview_bkg_texture', 'xview_syn_xview_bkg_color', 'xview_syn_xview_bkg_mixed']
    # # # # comments = ['xview_syn_xview_bkg_certain_models_texture', 'xview_syn_xview_bkg_certain_models_color', 'xview_syn_xview_bkg_certain_models_mixed']
    # # # # comments = ['xview_syn_xview_bkg_px20whr4_certain_models_texture', 'xview_syn_xview_bkg_px20whr4_certain_models_color', 'xview_syn_xview_bkg_px20whr4_certain_models_mixed']
    # comments = ['xview_syn_xview_bkg_px23whr4_scale_models_texture', 'xview_syn_xview_bkg_px23whr4_scale_models_color', 'xview_syn_xview_bkg_px23whr4_scale_models_mixed']
    # # comments = [ 'xview_syn_xview_bkg_px23whr4_small_models_color', 'xview_syn_xview_bkg_px23whr4_small_models_mixed']
    # base_cmt = 'px23whr4_seed{}'
    # # comments = [ 'xview_syn_xview_bkg_px23whr3_small_models_color', 'xview_syn_xview_bkg_px23whr3_small_models_mixed']
    # comments = ['xview_syn_xview_bkg_px23whr3_6groups_models_color', 'xview_syn_xview_bkg_px23whr3_6groups_models_mixed']
    # base_cmt = 'px23whr3_seed{}'
    # hyp_cmt = 'hgiou1_fitness'

    # hyp_cmt = 'hgiou1'
    # comments = ['xview_syn_xview_bkg_px23whr3_6groups_models_color', 'xview_syn_xview_bkg_px23whr3_6groups_models_mixed']
    # base_cmt = 'px23whr3_seed{}'
    # hyp_cmt = 'hgiou1_mean_best'
    # comments = ['xview_syn_xview_bkg_px23whr3_rnd_bwratio_models_color', 'xview_syn_xview_bkg_px23whr3_rnd_bwratio_models_mixed']
    # # base_cmt = 'px23whr3_seed{}'
    # # hyp_cmt = 'hgiou1_mean_best'
    # comments = ['xview_syn_xview_bkg_px23whr3_rnd_bwratio_flat0.8_models_color', 'xview_syn_xview_bkg_px23whr3_rnd_bwratio_flat0.8_models_mixed']
    # base_cmt = 'px23whr3_seed{}'
    # hyp_cmt = 'hgiou1_mean_best'
    # comments = ['xview_syn_xview_bkg_px23whr3_rnd_bwratio_asx_models_color', 'xview_syn_xview_bkg_px23whr3_rnd_bwratio_asx_models_mixed']
    # comments = ['xview_syn_xview_bkg_px23whr3_xratio_xcolor_models_color', 'xview_syn_xview_bkg_px23whr3_xratio_xcolor_models_mixed']
    # comments = ['xview_syn_xview_bkg_px23whr3_sbwratio_xratio_xcolor_models_color', 'xview_syn_xview_bkg_px23whr3_sbwratio_xratio_xcolor_models_mixed']
    # comments = ['xview_syn_xview_bkg_px23whr3_sbwratio_new_xratio_xcolor_models_color']
    # comments = ['xview_syn_xview_bkg_px23whr3_xbw_xcolor_model0_color', 'xview_syn_xview_bkg_px23whr3_xbw_xcolor_model0_mixed']
    # comments = ['xview_syn_xview_bkg_px23whr3_xbw_xrxc_spr_sml_models_color']#, 'xview_syn_xview_bkg_px23whr3_xbw_xrxc_spr_sml_models_mixed']
    # comments = ['xview_syn_xview_bkg_px23whr3_xbw_xcolor_model0_color', 'xview_syn_xview_bkg_px23whr3_xbw_xcolor_model0_mixed']
    # comments = ['xview_syn_xview_bkg_px15whr3_xbw_xcolor_model4_color', 'xview_syn_xview_bkg_px15whr3_xbw_xcolor_model4_mixed']
    # comments = ['xview_syn_xview_bkg_px23whr3_xbw_xcolor_gauss_model1_v1_color', 'xview_syn_xview_bkg_px23whr3_xbw_xcolor_gauss_model1_v1_mixed']
    # # # comments = ['xview_syn_xview_bkg_px23whr3_xbw_xcolor_model1_color', 'xview_syn_xview_bkg_px23whr3_xbw_xcolor_model1_mixed']
    # # # comments = ['xview_syn_xview_bkg_px15whr3_xbw_xcolor_model4_color', 'xview_syn_xview_bkg_px15whr3_xbw_xcolor_model4_mixed']
    # # # comments = ['xview_syn_xview_bkg_px15whr3_xbw_xcolor_model4_v1_color', 'xview_syn_xview_bkg_px15whr3_xbw_xcolor_model4_v1_mixed']
    # # # comments = ['xview_syn_xview_bkg_px15whr3_xbw_xcolor_model4_v2_color', 'xview_syn_xview_bkg_px15whr3_xbw_xcolor_model4_v1_mixed']
    # # comments = ['xview_syn_xview_bkg_px23whr3_xbw_xrxc_spr_sml_gauss_models_color', 'xview_syn_xview_bkg_px23whr3_xbw_xrxc_spr_sml_gauss_models_mixed']
    # prefix = 'xview + syn'
    # base_cmt = 'px23whr3_seed{}'
    # # hyp_cmt = 'hgiou1_mean_best'
    # sd=17
    # for cmt in comments:
    #     base_cmt = base_cmt.format(sd)
    #     opt = get_opt(comments=cmt)
    #
    #     cinx = cmt.find('model') # first letter index
    #     endstr = cmt[cinx:]
    #     rcinx = endstr.rfind('_')
    #     fstr = endstr[rcinx:] # '_' is included
    #     sstr = endstr[:rcinx]
    #     suffix = fstr + '_' + sstr
    #     opt.name = prefix + suffix
    #
    #     # prefix = 'xview + syn_'
    #     # lcmt = cmt.split('_')[-2:]
    #     # suffix = lcmt[1] + '_' + lcmt[0]
    #     # opt.name = prefix + suffix
    #     '''
    #     for a specified model id
    #     ********* manually change the IoU_threshold, iouv,0.5
    #     '''
    #     # hyp_cmt = 'hgiou1_1gpu'
    #     # opt.batch_size = 2
    #     # opt.model_id = int(lcmt[0][-1])
    #     # opt.conf_thres = 0.01
    #     #
    #     # opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_xview_with_model_{}_seed{}_1xSyn_miss'.format(hyp_cmt, sd))
    #     # all_weights = glob.glob(os.path.join(opt.weights_dir.format(opt.class_num, cmt, sd),  '*_{}_seed{}_1xSyn'.format(hyp_cmt, sd), 'best_seed{}_1xSyn.pt'.format(sd)))
    #     # opt.data = 'data_xview/{}_cls/{}/xviewtest_{}_with_model_m{}_miss.data'.format(opt.class_num, base_cmt, base_cmt, opt.model_id)
    #
    #     '''
    #     for whole validation dataset T_xview
    #     ** important : area is all images area
    #     ****** keep IoU_threshold,iouv, as 0.5
    #     '''
    #     # hyp_cmt = 'hgiou1_1gpu'
    #     # opt.conf_thres = 0.1
    #     # opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_xview_with_model_{}_seed{}_1xSyn'.format(hyp_cmt, sd))
    #     # all_weights = glob.glob(os.path.join(opt.weights_dir.format(opt.class_num, cmt, sd),  '*_{}_seed{}_1xSyn'.format(hyp_cmt, sd), 'best_seed{}_1xSyn.pt'.format(sd)))
    #     # opt.data = 'data_xview/{}_cls/{}_seed{}/{}_seed{}_1xSyn_with_model.data'.format(opt.class_num, cmt, sd, cmt, sd)
    #
    #     '''
    #     for whole validation dataset T_xview
    #     'hgiou1_x{}s{}'.format(batch_size-syn_batch_size, syn_batch_size)
    #     '''
    #     # hyp_cmt = 'hgiou1_x5s3'
    #     # # hyp_cmt = 'hgiou1_xbkgonly_x3s5'
    #     # # hyp_cmt = 'hgiou1_x7s1'
    #     # # hyp_cmt = 'hgiou1_x6s2'
    #     # # hyp_cmt = 'hgiou1_x4s4'e
    #     # opt.conf_thres = 0.1
    #     # opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_xview_with_model_{}_seed{}'.format(hyp_cmt, sd))
    #     # all_weights = glob.glob(os.path.join(opt.weights_dir.format(opt.class_num, cmt, sd),  '*_{}_seed{}'.format(hyp_cmt, sd), 'best_seed{}.pt'.format(sd)))
    #     # opt.data = 'data_xview/{}_cls/{}_seed{}/{}_seed{}_with_model.data'.format(opt.class_num, cmt, sd, cmt, sd)
    #
    #     '''
    #     for  specified model id T_xview_m*
    #     'hgiou1_x{}s{}'.format(batch_size-syn_batch_size, syn_batch_size)
    #     '''
    #     # hyp_cmt = 'hgiou1_x5s3'
    #     # opt.batch_size = 2
    #     # hyp_cmt = 'hgiou1_xbkgonly_x1s7'
    #     hyp_cmt = 'hgiou1_xbkgonly_x3s5'
    #     # hyp_cmt = 'hgiou1_xbkgonly_x2s6'
    #     opt.batch_size = 8
    #     sidx = cmt.split('model')[-1][0]
    #     opt.model_id = int(sidx)
    #     opt.conf_thres = 0.1
    #     opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_xview_with_model_{}_seed{}'.format(hyp_cmt, sd))
    #     all_weights = glob.glob(os.path.join(opt.weights_dir.format(opt.class_num, cmt, sd),  '*_{}_seed{}'.format(hyp_cmt, sd), 'best_seed{}.pt'.format(sd)))
    #     # opt.data = 'data_xview/{}_cls/{}/xviewtest_{}_with_model_m{}_miss.data'.format(opt.class_num, base_cmt, base_cmt, opt.model_id)
    #     opt.data = 'data_xview/{}_cls/{}/xviewtest_{}_with_model_m{}_only.data'.format(opt.class_num, base_cmt, base_cmt, opt.model_id)
    #
    #     print(opt.data)
    #     all_weights.sort()
    #     opt.weights = all_weights[-1]
    #     print(opt.weights)
    #
    #     if not os.path.exists(opt.result_dir):
    #         os.makedirs(opt.result_dir)
    #
    #     opt.base_dir = opt.base_dir.format(opt.class_num, base_cmt.format(sd))
    #     print(opt.base_dir)
    #     test(opt.cfg,
    #          opt.data,
    #          opt.weights,
    #          opt.batch_size,
    #          opt.img_size,
    #          opt.conf_thres,
    #          opt.nms_iou_thres,
    #          opt.save_json, opt=opt)

    '''
    test for xview_px6whr4_ng0_* with model
    '''
    # # # comments = ['px6whr4_ng0']
    # # # comments = ['px20whr4']
    # # comments = ['px23whr4']
    # # base_cmt = 'px23whr4_seed{}'
    # # hyp_cmt = 'hgiou1_fitness'
    # hyp_cmt = 'hgiou1_mean_best'
    # hyp_cmt = 'hgiou1_1gpu'
    # # # # # hyp_cmt = 'hgiou1_2gpus'
    # hyp_cmt = 'hgiou1_1gpu'
    # comments = ['px23whr3']
    # base_cmt = 'px23whr3_seed{}'
    # sd=17
    # for cmt in comments:
    #     opt = get_opt(comments=cmt)
    #     opt.batch_size = 2
    #     # opt.conf_thres = 0.1
    #
    #     opt.model_id = 1
    #     # opt.model_id = 4
    #     opt.conf_thres = 0.01
    #     # opt.conf_thres = 0.001
    #     # opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_xview_with_model_{}_seed{}_modelid'.format(hyp_cmt, sd))
    #     if opt.model_id is not None:
    #         opt.data = 'data_xview/{}_cls/{}_seed{}/xviewtest_{}_seed{}_with_model_m{}_miss.data'.format(opt.class_num, cmt, sd, cmt, sd, opt.model_id)
    #         opt.name = 'xview' + '_model{}'.format(opt.model_id)
    #         opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_xview_with_model_{}_seed{}_m{}_miss'.format(hyp_cmt, sd, opt.model_id))
    #     else:
    #         opt.data = 'data_xview/{}_cls/{}_seed{}/xview_{}_seed{}_with_model.data'.format(opt.class_num, cmt, sd, cmt, sd)
    #         opt.name = 'xview'
    #         opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_xview_with_model_{}_seed{}'.format(hyp_cmt, sd))
    #
    #     print(opt.data)
    #
    #     opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_xview_with_model_{}_seed{}'.format(hyp_cmt, sd))
    #     if not os.path.exists(opt.result_dir):
    #         os.mkdir(opt.result_dir)
    #     print(opt.weights_dir.format(opt.class_num, cmt, sd,  '*_{}_seed{}'.format(hyp_cmt, sd)))
    #     all_weights = glob.glob(os.path.join(opt.weights_dir.format(opt.class_num, cmt, sd),  '*_{}_seed{}'.format(hyp_cmt, sd), 'best_{}_seed{}.pt'.format(cmt, sd)))
    #     all_weights.sort()
    #     opt.weights = all_weights[-1]
    #     print(opt.weights)
    #
    #     opt.base_dir = opt.base_dir.format(opt.class_num, base_cmt.format(sd))
    #     test(opt.cfg,
    #          opt.data,
    #          opt.weights,
    #          opt.batch_size,
    #          opt.img_siz,
    #          opt.conf_thres,
    #          opt.nms_iou_thres,
    #          opt.save_json, opt=opt)


