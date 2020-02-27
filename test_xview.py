import argparse
import json

from torch.utils.data import DataLoader

from utils.parse_config_xview import *
from models_xview import *

from utils.datasets_xview import *
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


def get_opt():
    parser = argparse.ArgumentParser(prog='test.py')

    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-{}cls_syn.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data_xview/{}_cls/xview_{}_{}.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='weights/{}_cls/{}_{}/best_{}_{}.pt', help='path to weights file')

    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=608, help='inference size (pixels)')

    parser.add_argument('--class_num', type=int, default=1, help='class number')  # 60 6
    parser.add_argument('--label_dir', type=str, default='/media/lab/Yang/data/xView_YOLO/labels/', help='*.json path')
    parser.add_argument('--weights_dir', type=str, default='weights/{}_cls/{}_{}/', help='to save weights path')
    parser.add_argument('--result_dir', type=str, default='result_output/{}_cls/{}_{}/', help='to save result files path')
    parser.add_argument('--writer_dir', type=str, default='writer_output/{}_cls/{}_{}/', help='*events* path')
    parser.add_argument("--syn_ratio", type=float, default=0, help="ratio of synthetic data: 0 0.25, 0.5, 0.75, 1.0")
    parser.add_argument('--syn_display_type', type=str, default='syn', help='syn, syn_texture, syn_color')

    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    opt = parser.parse_args()
    # opt.save_json = opt.save_json or any([x in opt.data for x in ['xview.data']])

    opt.cfg = opt.cfg.format(opt.class_num)

    opt.save_json = opt.save_json or any([x in opt.data for x in ['xview_{}_{}.data'.format(opt.syn_display_type, opt.syn_ratio)]])
    opt.weights_dir = opt.weights_dir.format(opt.class_num, opt.syn_display_type, opt.syn_ratio)
    opt.writer_dir = opt.writer_dir.format(opt.class_num, opt.syn_display_type, opt.syn_ratio)
    opt.data = opt.data.format(opt.class_num, opt.syn_display_type, opt.syn_ratio)
    opt.result_dir = opt.result_dir .format(opt.class_num, opt.syn_display_type, opt.syn_ratio)
    opt.weights = opt.weights.format(opt.class_num, opt.syn_display_type, opt.syn_ratio, opt.syn_display_type, opt.syn_ratio)
    opt.label_dir = opt.label_dir + '{}/{}_cls/{}_{}/'.format(opt.img_size, opt.class_num, opt.syn_display_type, opt.syn_ratio)
    return opt


def get_val_imgid_by_name(image_name, opt=None):
    image_id_name_maps = json.load(
        open(os.path.join(opt.label_dir, 'all_image_ids_names_dict_{}cls.json'.format(opt.class_num))))
    img_ids = [int(k) for k in image_id_name_maps.keys()]
    img_names = [v for v in image_id_name_maps.values()]
    return img_ids[img_names.index(image_name)]


def test(cfg,
         data,
         weights=None,
         batch_size=16,
         img_size=416,
         conf_thres=0.001,
         iou_thres=0.5,  # for nms
         save_json=False,
         model=None,
         dataloader=None,
         opt=None):
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
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
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
    path = data['valid']  # path to test images
    lbl_path = data['valid_label']
    # path = data['valid_rare']  # path to test images
    # lbl_path = data['valid_rare_label']
    names = load_classes(data['names'])  # class names
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # for mAP@0.5
    niou = iouv.numel()

    # fixme
    result_json_file = 'results_{}_{}.json'.format(opt.syn_display_type, opt.syn_ratio)
    gt_json_file = 'xViewval_{}_{}cls_{}_{}_xtlytlwh.json'.format(img_size, opt.class_num, opt.syn_display_type, opt.syn_ratio)

    # result_json_file = 'results_rare.json'
    # gt_json_file = 'xViewval_rare_{}_{}cls_xtlytlwh.json'.format(img_size, opt.class_num)

    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels(path, lbl_path, img_size, batch_size, rect=True)  # , cache_labels=True
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
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        # print('targets--', targets.shape)
        # print('paths--', paths)
        # print('shapes', shapes)
        # print(targets)
        # exit(0)
        _, _, height, width = imgs.shape  # batch size, channels, height, width

        # Plot images with bounding boxes
        if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
            plot_images(imgs=imgs, targets=targets, paths=paths, fname='test_batch0.jpg')

        # Disable gradients
        with torch.no_grad():
            # Run model
            inf_out, train_out = model(imgs)  # inference and training outputs

            # Compute loss
            if hasattr(model, 'hyp'):  # if model has loss hyperparameters
                loss += compute_loss(train_out, targets, model)[1][:3].cpu()  # GIoU, obj, cls

            # Run NMS
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, 1), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            # fixme
            # clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                # fixme
                # print('paths[si]', paths[si]) #  /media/lab/Yang/data/xView_YOLO/images/608/1094_11.jpg
                # image_id = int(Path(paths[si]).stem.split('_')[-1])
                #fixme
                # image_name = paths[si].split('/')[-1]
                # image_id = get_val_imgid_by_name(image_name, opt)
                image_name = paths[si].split('/')[-1]
                image_id = si

                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner # xtlytlwh
                for di, d in enumerate(pred):
                    jdict.append({'image_id': image_id,
                                  'category_id': xview_classes[int(d[5])],
                                  'bbox': [floatn(x, 3) for x in box[di]],
                                  'score': floatn(d[4], 5)})  # conf

            # Assign all predictions as incorrect
            correct = torch.zeros(len(pred), niou, dtype=torch.bool)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * torch.Tensor([width, height, width, height]).to(device)

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices

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
            stats.append((correct, pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        # if niou > 1:
        #       p, r, ap, f1 = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # average across ious
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
        img_names = [x.split('/')[-1] for x in dataloader.dataset.img_files]
        #fixme
        # img_id_maps = json.load(
        #     open(os.path.join(opt.label_dir, 'all_image_ids_names_dict_{}cls.json'.format(opt.class_num))))
        # img_id_list = [k for k in img_id_maps.keys()]
        # img_name_list = [v for v in img_id_maps.values()]
        # imgIds = [img_id_list[img_name_list.index(v)] for v in img_name_list if
        #           v in img_names]  # note: index is the same as the keys
        # sids = set(imgIds)
        # print('imgIds', len(imgIds), 'sids', len(sids))

        # imgIds = [get_val_imgid_by_name(na) for na in img_names]
        # sids = set(imgIds)
        # print('imgIds', len(imgIds), 'sids', len(sids))
        imgIds = np.arange(len(output))

        with open(opt.result_dir + result_json_file, 'w') as file:
            # json.dump(jdict, file)
            json.dump(jdict, file, ensure_ascii=False, indent=2, cls=MyEncoder)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except:
            print('WARNING: missing pycocotools package, can not compute official COCO mAP. See requirements.txt.')

            # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            # fixme
            # cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
            cocoGt = COCO(opt.label_dir + gt_json_file)
            cocoDt = cocoGt.loadRes(opt.result_dir + result_json_file)  # initialize COCO pred api

            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss / len(dataloader)).tolist()), maps
    # return (mp, mr, map, mf1, *(loss / len(dataloader)).tolist())


if __name__ == '__main__':
    opt = get_opt()

    if opt.task == 'test':  # task = 'test', 'study', 'benchmark'
        # Test

        test(opt.cfg,
             opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json, opt=opt)

    elif opt.task == 'benchmark':
        # mAPs at 320-608 at conf 0.5 and 0.7
        y = []
        for i in [320, 416, 512, 608]:
            for j in [0.5, 0.7]:
                t = time.time()
                r = test(opt.cfg, opt.data, opt.weights, opt.batch_size, i, opt.conf_thres, j, opt.save_json)[0]
                y.append(r + (time.time() - t,))
        np.savetxt(opt.result_dir + 'benchmark.txt', y, fmt='%10.4g')  # y = np.loadtxt('study.txt')

    elif opt.task == 'study':
        # Parameter study
        y = []
        x = np.arange(0.4, 0.9, 0.05)
        for i in x:
            t = time.time()
            r = test(opt.cfg, opt.data, opt.weights, opt.batch_size, opt.img_size, opt.conf_thres, i, opt.save_json,
                     opt=opt)
            y.append(r + (time.time() - t,))
        np.savetxt(opt.result_dir + 'study.txt', y, fmt='%10.4g')  # y = np.loadtxt('study.txt')

        # Plot
        fig, ax = plt.subplots(3, 1, figsize=(6, 6))
        y = np.stack(y, 0)
        ax[0].plot(x, y[:, 2], marker='.', label='mAP@0.5')
        ax[0].set_ylabel('mAP')
        ax[1].plot(x, y[:, 3], marker='.', label='mAP@0.5:0.95')
        ax[1].set_ylabel('mAP')
        ax[2].plot(x, y[:, -1], marker='.', label='time')
        ax[2].set_ylabel('time (s)')
        for i in range(3):
            ax[i].legend()
            ax[i].set_xlabel('iou_thr')
        fig.tight_layout()
        plt.savefig(opt.result_dir + 'study.jpg', dpi=200)

    # from pycocotools.coco import COCO
    # from pycocotools.cocoeval import COCOeval
    # cocoGt = COCO(opt.label_dir + 'xViewval_{}_{}cls_xtlytlwh.json'.format(opt.img_size, opt.class_num))
    # cocoDt = cocoGt.loadRes(opt.result_dir + 'results.json')  # initialize COCO pred api
    # cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

    # img_names = [x.split('/')[-1] for x in dataloader.dataset.img_files]
    # img_id_maps = json.load(open(os.path.join(opt.label_dir, 'all_image_ids_names_dict_{}cls.json'.format(opt.class_num))))
    # img_id_list = [k for k in img_id_maps.keys()]
    # img_name_list = [v for v in img_id_maps.values()]
    # imgIds = [img_id_list[img_name_list.index(v)] for v in img_name_list if v in img_names] # note: index is the same as the keys
    # cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # cocoEval.summarize()
    # mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)
    #
    # # Return results
    # maps = np.zeros(nc) + map
    # for i, c in enumerate(ap_class):
    #     maps[c] = ap[i]
