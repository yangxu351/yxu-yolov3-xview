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
    last_num = 1
    iou_num = 7
    mp_arr = np.zeros((last_num, iou_num))
    mr_arr = np.zeros((last_num, iou_num))
    map_arr = np.zeros((last_num, iou_num))
    mf1_arr = np.zeros((last_num, iou_num))
    # Configure run
    data = parse_data_cfg(data)
    nc = int(data['classes'])  # number of classes
    # fixme--yang.xu
    path = data['test']  # path to test images
    lbl_path = data['test_label']
    names = load_classes(data['names'])  # class names
    m_key = 215
    for ix, mk in enumerate([m_key]):
   #  for ix, mk in enumerate(model_maps.keys()):
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

#        if apN == 20:
#            iouv = torch.linspace(0.2, 0.95, 10).to(device)  # iou vector for mAP@0.2:0.95
#        else:
        iouv = torch.linspace(0.2, 0.5, iou_num).to(device)  # iou vector for mAP@0.2:0.5
        #fixme --yang.xu
        # iouv = iouv[0].view(1)  # for mAP@0.5
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
        for ox in range(len(iouv)):
            iou = iou[ox].view(1)
            s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@{:.1f}'.format(iou[0]), 'F1')
            p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
            loss = torch.zeros(3)
            jdict, stats, ap, ap_class = [], [], [], []
            # fixme --yang.xu
            sum_labels = 0
            for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
                imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
                targets = targets.to(device)
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

                        # Search for detections
                        if len(pi) and len(ti) and not len(ni):
                            ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices
                            # print('ious ', ious.shape)
                            # Append detections
                            for j in (ious > iou).nonzero():
                                d = ti[i[j]]  # detected target
                                # print('d', d)
                                if d not in detected:
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iou  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break
                        elif len(pi) and not len(ti) and len(ni):
                            neu_ious, nix = box_iou(pred[pi, :4], tbox[ni]).max(1)
                            for s in (neu_ious > iou).nonzero():
    #                            print('s --------> ', s)
                                d = ni[nix[s]]  # detected target
                                if d not in detected:
                                    detected.append(d)
                                    neu_correct[pi[s]] = neu_ious[s] > iou  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break
                        elif len(pi) and len(ti) and len(ni):
                            ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices
                            neu_ious, nix = box_iou(pred[pi, :4], tbox[ni]).max(1)
                            for j in (ious > iou).nonzero():
                                d = ti[i[j]]  # detected target
                                if d not in detected:
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iou  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break
                            for s in (neu_ious > iou).nonzero():
    #                            print('s --------> ', s)
                                d = ni[nix[s]]  # detected neutral target
                                if d not in detected:
                                    detected.append(d)
                                    neu_correct[pi[s]] = neu_ious[s] > iou  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break
#                    print('detected ', detected)
                    # Append statistics (correct, conf, pcls, tcls, neu_correct)
                    # pred (x1, y1, x2, y2, object_conf, conf, class)
                    stats.append((correct[:], pred[:, 4].cpu(), pred[:, 5].cpu(), tcls, neu_correct[:]))
#                if len(tcls) and len(correct):
#                    print('\n correct: {}  pred[:,4]:{}  pred[:, 5]:{} tcls:{}'.format(correct, pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
                # if len(tcls) and len(neu_correct):
#                    print('\n neu_correct: {}  pred[:,4]:{}  pred[:, 5]:{} tcls:{}'.format(neu_correct, pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

                    print('sum all labels', sum_labels)
                    # Compute statistics
                    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
                    if len(stats):
                        pr_name= opt.name # + ' @IoU: {:.2f} '.format(iouv[0]) + ' conf_thres: {} '.format(conf_thres)
                        pr_legend = opt.legend
                        # print('*stats', *stats)
                        p, r, ap, f1, ap_class = ap_per_class(*stats, pr_path=opt.result_dir, pr_name= pr_name, pr_legend=pr_legend, rare_class=opt.rare_class, apN=opt.apN)

                        print('dataset.batch ', dataset.batch.shape)
                        # exit(0)
                        area = (img_size*opt.res)*(img_size*opt.res)*dataset.batch.shape[0]*1e-6

                        plot_roc_easy_hard(*stats, pr_path=opt.result_dir, pr_name= pr_name, pr_legend=pr_legend, rare_class=opt.rare_class, area=area, ehtype=opt.type, title_data_name='xview')
                        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
                        #fixme --yang.xu compute before
                        # nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
                        st3 = stats[3][stats[3] == opt.rare_class]
                        nt = np.bincount(st3.astype(np.int64), minlength=nc)  # number of targets per class
                    else:
                        nt = torch.zeros(1)

            #fixme--yang.xu
            mp_arr[ix, ox] = mp
            mr_arr[ix, ox] = mr
            map_arr[ix, ox] = map
            mf1_arr[ix, ox] = mf1

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
                result_json_file = 'results_{}_{}.json'.format(opt.name, mk)
                with open(os.path.join(opt.result_dir, result_json_file), 'w') as file:
                    # json.dump(jdict, file)
                    json.dump(jdict, file, ensure_ascii=False, indent=2, cls=MyEncoder)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp_arr.mean(axis=1), mr_arr.mean(axis=1), map_arr.mean(axis=1), mf1_arr.mean(axis=1)))
    return seen, nt.sum(), mp_arr.mean(axis=1), mr_arr.mean(axis=1), map_arr.mean(axis=1), mf1_arr.mean(axis=1)


def get_opt(dt=None, sr=None, comments=''):
    parser = argparse.ArgumentParser() # prog="test.py"

    parser.add_argument("--cfg", type=str, default="cfg/yolov3-spp-{}cls_syn.cfg", help="*.cfg path")
    parser.add_argument("--data", type=str, default="data_xview/{}_cls/{}/xview_{}_{}.data", help="*.data path")
    parser.add_argument("--weights", type=str, default="weights/{}_cls/{}_{}/best_{}_{}.pt", help="path to weights file")

    parser.add_argument("--batch-size", type=int, default=8, help="size of each image batch") # 2
    parser.add_argument("--img_size", type=int, default=608, help="inference size (pixels)")
    parser.add_argument("--res", type=float, default=0.3, help="resolution")
    parser.add_argument("--margin", type=int, default=30, help="margin size (pixels)")

    parser.add_argument("--class_num", type=int, default=1, help="class number")  # 60 6
    parser.add_argument("--label_dir", type=str, default="/media/lab/Yang/data/xView_YOLO/labels/", help="*.json path")
    parser.add_argument("--weights_dir", type=str, default="weights/{}_cls/{}_seed{}/", help="to save weights path")
    parser.add_argument("--result_dir", type=str, default="result_output/{}_cls/{}_seed{}/{}/", help="to save result files path")
    parser.add_argument("--grids_dir", type=str, default="grids_dir/{}_cls/{}_seed{}/", help="to save grids images")
    parser.add_argument("--syn_ratio", type=float, default=sr, help="ratio of synthetic data: 0 0.25, 0.5, 0.75")
    parser.add_argument("--syn_display_type", type=str, default=dt, help="syn_texture0, syn_color0, syn_texture, syn_color, syn_mixed, syn")
    parser.add_argument("--base_dir", type=str, default="data_xview/{}_cls/{}/", help="without syn data path")

    parser.add_argument("--conf-thres", type=float, default=0.01, help="0.1 0.05 0.001 object confidence threshold")
    parser.add_argument("--nms-iou-thres", type=float, default=0.5, help="NMS 0.5  0.6 IOU threshold for NMS")
    parser.add_argument("--save_json", action="store_true", default=True, help="save a cocoapi-compatible JSON results file")
    parser.add_argument("--task", default="test", help="test study benchmark")
    parser.add_argument("--device", default="1", help="device id (i.e. 0 or 0,1) or cpu")
    parser.add_argument("--name", default='', help="file name")
    parser.add_argument("--legend", default='', help="figure legend")
    parser.add_argument("--cmt", default=comments, help="comments")
    parser.add_argument("--model_id", type=int, default=None, help="specified model id")
    parser.add_argument("--rare_class", type=int, default=None, help="specified rare class")
    parser.add_argument("--type", default="hard", help="hard, easy")
    # parser.add_argument("--apN", type=int, default=50, help="ap@N 50, 40, 20")
    opt = parser.parse_args()
    # opt.save_json = opt.save_json or any([x in opt.data for x in ["xview.data"]])
    opt.cfg = opt.cfg.format(opt.class_num)

    return opt


def main():
    base_cmt = "px23whr3_seed{}"
    # hyp_cmt = "hgiou1_1gpu"

    hyp_cmt = "hgiou1_1gpu_val_syn"
    prefix = 'syn_map'

    px_thres = 23
    whr_thres = 3 # 4
    sd = 17
    model_ids = [4, 1, 5, 5, 5]
    rare_classes = [1, 2, 3, 4, 5]
    ap_list = [20, 40, 50]
    eh_types = ["hard", "easy"]

    for typ in eh_types:
#        df_pr_ap = pd.DataFrame(columns=["Version", "Seen", "NT", "Precision", "Recall", "AP@{}".format(apN), "F1"])
#         df_pr_ap = pd.DataFrame(columns=["Version","Seen", "NT", "mAP(0.2:0.5:0.05)".format(ap_list[0]), "AP{}".format(ap_list[1]), "AP{}".format(ap_list[2])])
        df_pr_ap = pd.DataFrame(columns=["Version","Seen", "NT", "mAP(0.2:0.5:0.05)"])
        for ix, cmt in enumerate(comments):
            VN = ix + base_version
            base_cmt = base_cmt.format(sd)
            opt = get_opt(comments=cmt)
            opt.device = "2"

            cinx = cmt.find('_RC') # first letter index
            endstr = cmt[cinx:]
            rcinx = endstr.rfind('_') # _RC*_v*
            sstr = endstr[:rcinx]
            if cinx >= 0:
                medix = cmt.find('_dyn')
                mstr =  cmt[medix:cinx] # _dyn*_color_bias*
                suffix = sstr + '_mAP(0.2:0.5)'
            else:
                mstr = ''
                suffix = ''

            opt.legend = prefix + suffix
            opt.name = prefix + mstr + suffix # 'syn_RC*_v*_dyn*_color_bias*_AP*

            ''' for specified model id '''
            opt.batch_size = 8
            opt.rare_class = int(cmt[cinx+3])
            opt.model_id = model_ids[rare_classes.index(opt.rare_class)]
            print("opt.model_id", opt.model_id, 'opt.rare_class ', opt.rare_class)

            opt.conf_thres = 0.01
            tif_name = "xview"
            ############# 2 images test set
#            opt.type = "easy"
#            opt.type = "hard"
            opt.type = typ

            opt.name += "_{}".format(opt.type)
#            opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, "test_on_xview_{}_m{}_rc{}_{}".format(hyp_cmt, opt.model_id, opt.rare_class, opt.type))
            opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, "test_on_xview_{}_m{}_rc{}_ap{}_{}".format(hyp_cmt, opt.model_id, opt.rare_class, apN, opt.type))
#            opt.data = "data_xview/{}_cls/{}/xviewtest_{}_m{}_rc{}_{}.data".format(opt.class_num, base_cmt, base_cmt, opt.model_id, opt.rare_class, opt.type)
            opt.data = "data_xview/{}_cls/{}/xview_rc_test_{}_m{}_rc{}_{}.data".format(opt.class_num, base_cmt, base_cmt, opt.model_id, opt.rare_class, opt.type)

            if not os.path.exists(opt.result_dir):
                os.makedirs(opt.result_dir)
#            print(os.path.join(opt.weights_dir.format(opt.class_num, cmt, sd), "*_{}_seed{}".format(hyp_cmt, sd), "best_seed{}.pt".format(sd)))
#            print(glob.glob(os.path.join(opt.weights_dir.format(opt.class_num, cmt, sd), "*_{}_seed{}".format(hyp_cmt, sd), "best_seed{}.pt".format(sd))))
            all_weights = glob.glob(os.path.join(opt.weights_dir.format(opt.class_num, cmt, sd), "*_{}_seed{}".format(hyp_cmt, sd), "best_*seed{}.pt".format(sd)))
            all_weights.sort()
            opt.weights = all_weights[-1]

            print(opt.weights)
            print(opt.data)


            seen, nt, mp, mr, mapv, mf1 = test(opt.cfg,
                 opt.data,
                 opt.weights,
                 opt.batch_size,
                 opt.img_size,
                 opt.conf_thres,
                 opt.nms_iou_thres,
                 opt.save_json, opt=opt)

            df_pr_ap.at[ix, "Version"] = VN
            df_pr_ap.at[ix, "Seen".format(apN)] = seen
            df_pr_ap.at[ix, "NT".format(apN)] = nt
            df_pr_ap.at[ix, "mAP(0.2:0.5:0.05)".format(apN)] = mapv

        sinx = cmt.find('dyn')
        einx = cmt.find('bias')+4
        dynstr = cmt[sinx:einx]

        csv_dir = "result_output/{}_cls/{}/".format(opt.class_num, cmt[:einx] + '_RC' + str(opt.rare_class))
        if not os.path.exists(csv_dir):
            os.mkdir(csv_dir)
#        csv_name =  "{}_{}_RC{}_AP{}_{}.xls".format(prefix, dynstr, opt.rare_class, apN, opt.type)
#        df_pr_ap.to_excel(os.path.join(csv_dir, csv_name), index=False)


        csv_name =  "{}_{}_RC{}_{}.xlsx".format(prefix, dynstr, opt.rare_class, opt.type)
#            if os.path.exists(os.path.join(csv_dir, csv_name)):
#                mode = 'a'
#            else:
#                mode = 'w'
#         mode = 'w'
#         with pd.ExcelWriter(os.path.join(csv_dir, csv_name), mode=mode) as writer:
#             df_pr_ap.to_excel(writer, sheet_name='RC{}_{}'.format(opt.rare_class, opt.type), index=False) #
        mode = 'w'
        with pd.ExcelWriter(os.path.join(csv_dir, csv_name), mode=mode) as writer:
            df_pr_ap.to_excel(writer, sheet_name='RC{}_{}'.format(opt.rare_class, opt.type), index=False) #



if __name__ == "__main__":

    '''
    test for syn_xveiw_background_*_with_model
    '''
    # #
#
#    model_id = 5
#    rare_cls = 5
#
#    apN = 20
#    apN = 40
#    apN = 50

    ############### dynmu color
    '''RC1'''
    comments = []
    pros = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    base_version = 50
    for ix, pro in enumerate(pros):
        cmt = 'syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias{}_RC1_v{}_color'.format(pro, ix+ base_version)
        comments.append(cmt)
    main()

    '''RC2'''
#     comments = []
#     pros = [0.5, -0.5, -1.5, -2.5, -3.5, -4.5]
#     base_version = 30
#     for ix, pro in enumerate(pros):
#         cmt = 'syn_xview_bkg_px23whr3_xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias{}_RC2_v{}_color'.format(pro, ix+ base_version)
#         comments.append(cmt)
#     main()
#
#     '''RC3'''
#     comments = []
#     pros = [0.5, -0.5, -1.5, -2.5, -3.5, -4.5]
#     base_version = 30
#     for ix, pro in enumerate(pros):
#         cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias{}_RC3_v{}_color'.format(pro, ix+ base_version)
#         comments.append(cmt)
#     main()
#
#     '''RC4'''
#     comments = []
#     pros = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
#     base_version = 30
#     for ix, pro in enumerate(pros):
#         cmt = 'syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias{}_RC4_v{}_color'.format(pro, ix+ base_version)
#         comments.append(cmt)
#     main()
#
#     '''RC5'''
#     comments = []
#     pros = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
#     base_version = 20
#     for ix, pro in enumerate(pros):
#         cmt = 'syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynmu_color_bias{}_RC5_v{}_color'.format(pro, ix+ base_version)
#         comments.append(cmt)
#     main()
#
#     ###########################################
#
#     ################ dynsigma color
#
#     '''RC1'''
#     comments = []
#     pros = [0, 0.2, 0.4, 0.6, 0.8]
# #    base_version = 60
# #    for ix, pro in enumerate(pros):
# #        cmt = 'syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_dynsigma_color_bias{}_RC1_v{}_color'.format(pro, ix+ base_version)
# #        comments.append(cmt)
#     base_version = 80
#     for ix, pro in enumerate(pros):
#         cmt = 'syn_xview_bkg_px15whr3_xbw_xbkg_unif_mig21_shdw_split_scatter_gauss_rndsolar_light_dynsigma_color_bias{}_RC1_v{}_color'.format(pro, ix+ base_version)
#         comments.append(cmt)
#     main()
#
#     '''RC2'''
#     comments = []
#     pros = [0, 0.2, 0.4, 0.6, 0.8]
#     base_version =40
#     for ix, pro in enumerate(pros):
#         cmt = 'syn_xview_bkg_px23whr3_xbsw_xwing_xbkg_shdw_split_scatter_gauss_rndsolar_dynsigma_color_bias{}_RC2_v{}_color'.format(pro, ix+ base_version)
#         comments.append(cmt)
#     main()
#
#     '''RC3'''
#     comments = []
#     pros = [0, 0.2, 0.4, 0.6, 0.8]
#     base_version = 40
#     for ix, pro in enumerate(pros):
#         cmt = 'syn_xview_bkg_px23whr3_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynsigma_color_bias{}_RC3_v{}_color'.format(pro, ix+ base_version)
#         comments.append(cmt)
#     main()
#
#     '''RC4'''
#     comments = []
#     pros = [0, 0.2, 0.4, 0.6, 0.8]
#     base_version = 40
#     for ix, pro in enumerate(pros):
#         cmt = 'syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynsigma_color_bias{}_RC4_v{}_color'.format(pro, ix+ base_version)
#         comments.append(cmt)
#     main()
#
#     '''RC5'''
#     comments = []
#     pros = [0, 0.2, 0.4, 0.6, 0.8]
#     base_version = 30
#     for ix, pro in enumerate(pros):
#         cmt = 'syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_split_scatter_gauss_rndsolar_dynsigma_color_bias{}_RC5_v{}_color'.format(pro, ix+ base_version)
#         comments.append(cmt)
#     main()

    ###########################################




