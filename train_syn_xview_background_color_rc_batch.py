import os
import sys
import time
import argparse
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import test_xview as test  # import test.py to get mAP after each epoch
from models_xview import *
from utils.datasets_xview import *
from utils.utils_xview import *
from utils.torch_utils import *
import warnings

warnings.filterwarnings("ignore")



#fixme before git pull at April 23
# Hyperparameters https://github.com/ultralytics/yolov3/issues/310
# hyp = {'giou': 1.0, #1.0,  1.5# giou loss gain 3.54
#        'cls': 37.4,  # cls loss gain
#        'cls_pw': 1.0,  # cls BCELoss positive_weight
#        'obj': 49.5, # 49.5,  # obj loss gain (*=img_size/320 if img_size != 320)
#        'obj_pw': 1.0,  # obj BCELoss positive_weight
#        'iou_t': 0.225,  # iou training threshold
#        'lr0': 0.00579,  # initial learning rate (SGD=1E-3, Adam=9E-5)
#        'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
#        'momentum': 0.937,  # SGD momentum
#        'weight_decay': 0.000484,  # optimizer weight decay
#        'fl_gamma': 0.5,  # focal loss gamma
#        'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
#        'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
#        'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
#        'degrees': 1.98,  # image rotation (+/- deg)
#        'translate': 0.05,  # image translation (+/- fraction)
#        'scale': 0.05,  # image scale (+/- gain)
#        'shear': 0.641}  # image shear (+/- deg)

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
def infi_loop(dl):
    while True:
        for (imgs, targets, paths, _) in dl:
            yield imgs, targets, paths


def train(opt):
    cfg = opt.cfg
    data = opt.data
    img_size = opt.img_size
    epochs = 1 if opt.prebias else opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    # accumulate = max(round(64 / batch_size), 1)
    weights = opt.weights  # initial training weights

    mixed_precision = True
    try:  # Mixed precision training https://github.com/NVIDIA/apex
        from apex import amp
    except:
        mixed_precision = False  # not installed
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    print('device ', device)
    # exit(0)
    print('hyp_cmt_name', hyp_cmt_name)

    if device.type == 'cpu':
        mixed_precision = False

    tb_writer = None
    try:
        # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=opt.writer_dir)
    except:
        print('SummaryWriter error')
        return

    # Initialize
    init_seeds()
    if opt.multi_scale:
        img_sz_min = round(img_size / 32 / 1.5)
        img_sz_max = round(img_size / 32 * 1.5)
        img_size = img_sz_max * 32  # initiate with maximum multi_scale size
        print('Using multi-scale %g - %g' % (img_sz_min * 32, img_size))


    # Configure run
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    train_label_path = data_dict['train_label']
    test_label_path = data_dict['valid_label']
    nc = int(data_dict['classes'])  # number of classes
    syn_0_xview_number = data_dict['syn_0_xview_number']
    loop_count = int(syn_0_xview_number) // batch_size

    # Remove previous results
    for f in glob.glob('trn_patch_images/*_batch*.jpg') + glob.glob(results_file):
        os.remove(f)

    # Initialize model
    model = Darknet(cfg, arc=opt.arc).to(device)

    # Optimizer
    pg0, pg1 = [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if 'Conv2d.weight' in k:
            pg1 += [v]  # parameter group 1 (apply weight_decay)
        else:
            pg0 += [v]  # parameter group 0

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    del pg0, pg1

    # https://github.com/alphadl/lookahead.pytorch
    # optimizer = torch_utils.Lookahead(optimizer, k=5, alpha=0.5)

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    #fixme
    best_fitness = 0.0
    best_5_ckpt = {}
    # attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        chkpt = torch.load(weights, map_location=device)

        # load model
        try:
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        # load optimizer
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

        # load results
        if chkpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        start_epoch = chkpt['epoch'] + 1
        del chkpt

    elif len(weights) > 0:  # darknet format
        # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        cutoff = load_darknet_weights(model, weights)

    if opt.prebias:
        # Update params (bias-only training allows more aggressive settings: i.e. SGD ~0.1 lr0, ~0.9 momentum)
        for p in optimizer.param_groups:
            p['lr'] = 0.1  # learning rate
            if p.get('momentum') is not None:  # for SGD but not Adam
                p['momentum'] = 0.9

        for name, p in model.named_parameters():
            p.requires_grad = True if name.endswith('.bias') else False
    #fixme -- for x in [0.8, 0.9]
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in  [0.8, 0.9]], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Initialize distributed training
    #fixme --yang.xu Do not need distribution
    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # if device.type != 'cpu' and torch.cuda.device_count() > 1:
    #     # model = nn.DataParallel(model)
    #     model = nn.parallel.DataParallel(model, device_ids=[0, 1])
    #     model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataset
    dataset = LoadImagesAndLabels(train_path, train_label_path, img_size, batch_size,
                                  class_num=opt.class_num,
                                  augment=True,  # False, #True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  image_weights=False,
                                  cache_labels=epochs > 10,
                                  cache_images=opt.cache_images and not opt.prebias)

    # Dataloader
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)
    # Test Dataloader
    if not opt.prebias:
        testloader = torch.utils.data.DataLoader(
            LoadImagesAndLabels(test_path, test_label_path, opt.img_size, batch_size * 2, class_num=opt.class_num,
                                hyp=hyp,
                                rect=True,
                                cache_labels=True,
                                cache_images=opt.cache_images, with_modelid=False),
            batch_size=batch_size * 2,
            num_workers=nw,
            pin_memory=True,
            collate_fn=dataset.collate_fn)

    # Start training
    # fixme
    # nb = len(dataloader)
    nb = loop_count
    print('nb ', nb)
    model.nc = nc  # attach number of classes to model
    model.arc = opt.arc  # attach yolo architecture
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    maps = np.zeros(nc)  # mAP per class

    #fixme --yang.xu
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    #fixme --yang.xu
    # Model EMA
    # ema = torch_utils.ModelEMA(model)

    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    #fixme --yang.xu
    # torch_utils.model_info(model, report='summary')  # 'full' or 'summary'
    torch_utils.model_info(model)

    print('Using %g dataloader workers' % nw)
    print('Starting %s for %g epochs...' % ('prebias' if opt.prebias else 'training', epochs))
    trn_names_dict = {}
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        # if epoch == epochs-1:
        #     return
        model.train()
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))

        # Update image weights (optional)
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(4).to(device)  # mean losses
        #fixme
        # pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        # for i, (imgs, targets, paths, _) in pbar:
        if epoch < 20:
            trn_names_dict[epoch] = []
        gen_data = infi_loop(dataloader)
        for i in range(nb):
            imgs, targets, paths = next(gen_data)
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)
            # print('targets ', targets.shape, targets)

            # Multi-Scale training
            if opt.multi_scale:
                if ni / accumulate % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                    img_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
            
            if epoch < 20:
                # print('i', i, 'epoch', epoch)
                for pi in range(batch_size):
                    trn_names_dict[epoch].append(Path(paths[pi]).name)

            # Plot images with bounding boxes
            if ni == 0:
                fname = 'trn_patch_images/train_batch%g.jpg' % i
                plot_images(imgs=imgs, targets=targets, paths=paths, fname=fname)
                if tb_writer:
                    tb_writer.add_image(fname, cv2.imread(fname)[:, :, ::-1], dataformats='HWC')
            # Run model
            pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Scale loss by nominal batch_size of 64
            loss *= batch_size / 64

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Accumulate gradient for x batches before optimizing
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                #fixme
                # ema.update(model)

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, len(targets), img_size)
            print(s)
            # pbar.set_description(s)
            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        scheduler.step()
        print(scheduler.get_lr())
        # #fixme ---
        if tb_writer:
            tb_writer.add_scalar('lr', np.array(scheduler.get_lr())[0], epoch)
            # tb_writer.add_graph(model,imgs)
        #fixme ---yang.xu
        # ema.update_attr(model)
        
        chkpt = {'epoch': epoch,
                   #fixme --yang.xu
                   'model': model.module.state_dict() if type(
                       model) is nn.parallel.DistributedDataParallel else model.state_dict()}

        # Save last checkpoint
        # last_before = last.replace('.pt', '_before.pt')
        # torch.save(chkpt, last_before)
        
        # Process epoch results
        final_epoch = epoch + 1 == epochs
        if opt.prebias:
            print_model_biases(model)
        elif not opt.notest or final_epoch:  # Calculate mAP
            #fixme

            results, maps = test.test(cfg,
                                      data,
                                      batch_size=batch_size * 2,
                                      img_size=opt.img_size,
                                      conf_thres= opt.conf_thres, # 0.1, # 0.001 if final_epoch else 0.1,  # 0.1 for speed
                                      nms_iou_thres= opt.nms_iou_thres, # 0.5, # 0.6 if final_epoch and is_xview else 0.5,
                                      save_json=True,  # final_epoch and is_xview, #fixme
                                      model=model,#fixme
                                      # model=ema.ema,
                                      dataloader=testloader, opt=opt)

        # Write epoch results
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        if len(opt.name) and opt.bucket and not opt.prebias:
            os.system('gsutil cp results.txt gs://%s/results%s.txt' % (opt.bucket, opt.name))

        # Write Tensorboard results
        #fixme --yang.xu
        # if tb_writer:
        #     x = list(mloss) + list(results)
        #     titles = ['GIoU', 'Objectness', 'Classification', 'Train loss',
        #               'Precision', 'Recall', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification']
        #     for xi, title in zip(x, titles):
        #         tb_writer.add_scalar(title, xi, epoch)
        if tb_writer:
            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/F1',
                    'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
            for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                tb_writer.add_scalar(tag, x, epoch)
        # Update best mAP
        #fixme--yang.xu
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save training results
        save = (not opt.nosave) or (final_epoch and not opt.evolve) or opt.prebias
        if save:
            with open(results_file, 'r') as f:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': f.read(),
                         #fixme --yang.xu
                         'model': model.module.state_dict() if type(
                             model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                         # 'model': ema.ema.module.state_dict() if hasattr(model, 'module') else ema.ema.state_dict(),
                         'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last checkpoint
            torch.save(chkpt, last)

            # Save best checkpoint
            #fixme
            # if best_fitness == fi and not final_epoch:
            #     torch.save(chkpt, best)

            if epoch >= epochs - 5:
                best_5_ckpt[epoch] = chkpt
                torch.save(best_5_ckpt, best)
            # Save backup every 10 epochs (optional)
            #fixme
            # if (epoch > 0 and epoch % 10 == 0):
            if (epoch > 0 and epoch % 50 == 0) :
                torch.save(chkpt, opt.weights_dir + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt

        # end epoch ----------------------------------------------------------------------------------------------------
    json.dump(trn_names_dict, open('input_trn_files/{}_{}_trn_names_of_20_epochs.json'.format(opt.name, hyp_cmt_name), 'w'), ensure_ascii=False, indent=2, cls=MyEncoder) 
    # png_name = 'results_{}_{}.png'.format(opt.syn_display_type, opt.syn_ratio)
    if tb_writer:
        tb_writer.close()
    png_name = 'results_{}.png'.format(opt.name)
    plot_results(result_dir=opt.result_dir, png_name=png_name, class_num=opt.class_num, title=opt.name)  # save as results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    #fixme --yang.xu
    print('dist destroy --begin')
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    print('dist destroy --end')
    torch.cuda.empty_cache()

    return results


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=17, help='seed')
    parser.add_argument('--cfg_dict', type=str, default='',
                        help='train_cfg/train_1cls_syn_only_mean_best_gpu0.json')
    parser.add_argument('--data', type=str, default='', help='*.data path')
    parser.add_argument('--epochs', type=int, default=220)  # 220 180 250  500200 batches at bs 16, 117263 images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=8)  # effective bs = batch_size * accumulate = 16 * 4 = 64

    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--img_size', type=int, default=608, help='inference size (pixels)')  # 416 608
    parser.add_argument('--class_num', type=int, default=1, help='class number')  # 60 6 1
    parser.add_argument('--model_id', type=int, default=None, help='model id')

    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-{}cls_syn.cfg', help='*.cfg path')
    parser.add_argument('--writer_dir', type=str, default='writer_output/{}_cls/{}_seed{}/{}/', help='*events* path')
    parser.add_argument('--weights_dir', type=str, default='weights/{}_cls/{}_seed{}/{}/', help='to save weights path')
    parser.add_argument('--result_dir', type=str, default='result_output/{}_cls/{}_seed{}/{}/', help='to save result files path')
    parser.add_argument('--base_dir', type=str, default='data_xview/{}_cls/{}/', help='without syn data path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')

    parser.add_argument('--accumulate', type=int, default=4, help='batches to accumulate before optimizing')
    parser.add_argument('--multi_scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--conf_thres', type=float, default=0.01, help='0.001 object confidence threshold')
    parser.add_argument('--nms_iou_thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--save_json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='', help="'test', 'study', 'benchmark'")

    parser.add_argument('--rect', default=False, action='store_true', help='rectangular training')
    parser.add_argument('--resume', default=False, action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache_images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='', help='initial weights')  # weights/ultralytics68.pt
    parser.add_argument('--arc', type=str, default='default', help='yolo architecture')  # defaultpw, uCE, uBCE
    parser.add_argument('--prebias', action='store_true', help='pretrain model biases')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--var', type=float, help='debug variable')
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    # main()
    # seeds = [17]#, 5, 9, 1024, 3] #   5, 9, 1024,
    # comments = ['syn_xview_background_color', 'syn_xview_background_mixed']
    # comments = ['xview_syn_xview_bkg_texture', 'xview_syn_xview_bkg_color', 'xview_syn_xview_bkg_mixed']
    # syn_ratios = [1, 2]
    # comments = ['px6whr4_ng0']
    # syn_ratios = [0]
    # comments = ['px23whr4']
    # syn_ratios = [0]
    # comments = ['syn_xview_bkg_certain_models_texture', 'syn_xview_bkg_certain_models_color', 'syn_xview_bkg_certain_models_mixed']
    # syn_ratios = [None]
    # comments = ['xview_syn_xview_bkg_certain_models_texture', 'xview_syn_xview_bkg_certain_models_color', 'xview_syn_xview_bkg_certain_models_mixed']
    # syn_ratios = [2]
    # comments = ['syn_xview_bkg_px20whr4_certain_models_texture', 'syn_xview_bkg_px20whr4_certain_models_color', 'syn_xview_bkg_px20whr4_certain_models_mixed']
    # syn_ratios = [None]
    # comments = ['xview_syn_xview_bkg_px20whr4_certain_models_texture', 'xview_syn_xview_bkg_px20whr4_certain_models_color', 'xview_syn_xview_bkg_px20whr4_certain_models_mixed']
    # syn_ratios = [1]
    # comments = ['syn_xview_bkg_px23whr4_scale_models_texture', 'syn_xview_bkg_px23whr4_scale_models_color', 'syn_xview_bkg_px23whr4_scale_models_mixed']
    # syn_ratios = [None]
    # comments = ['xview_syn_xview_bkg_px23whr4_scale_models_texture', 'xview_syn_xview_bkg_px23whr4_scale_models_color', 'xview_syn_xview_bkg_px23whr4_scale_models_mixed']
    # syn_ratios = [1]
    # comments = ['xview_syn_xview_bkg_px23whr4_small_models_color', 'xview_syn_xview_bkg_px23whr4_small_models_mixed']
    # syn_ratios = [1]
    # comments = ['syn_xview_bkg_px23whr4_small_models_color', 'syn_xview_bkg_px23whr4_small_models_mixed']
    # syn_ratios = [None]
    # comments = ['syn_xview_bkg_px23whr4_small_fw_models_color', 'syn_xview_bkg_px23whr4_small_fw_models_mixed']
    # syn_ratios = [None]
    # comments = ['px23whr3']
    # syn_ratios = [0]
    # comments = ['px23whr3']
    # syn_ratios = [0]
    # comments = ['syn_xview_bkg_px23whr3_small_models_color', 'syn_xview_bkg_px23whr3_small_models_mixed']
    # syn_ratios = [None]
    # comments = ['xview_syn_xview_bkg_px23whr3_small_models_color', 'xview_syn_xview_bkg_px23whr3_small_models_mixed']
    # syn_ratios = [1]
    # comments = ['syn_xview_bkg_px23whr3_6groups_models_color', 'syn_xview_bkg_px23whr3_6groups_models_mixed'] #
    # syn_ratios = [None, None]
    # comments = ['syn_xview_bkg_px23whr3_rnd_bwratio_models_color', 'syn_xview_bkg_px23whr3_rnd_bwratio_models_mixed'] #
    # syn_ratios = [None, None]
    # comments = ['xview_syn_xview_bkg_px23whr3_small_models_color', 'xview_syn_xview_bkg_px23whr3_small_models_mixed', 'px23whr3']
    # comments = ['xview_syn_xview_bkg_px23whr3_6groups_models_color','xview_syn_xview_bkg_px23whr3_6groups_models_mixed']
    # syn_ratios = [ 1, 1]
    # comments = ['px23whr3']
    # syn_ratios = [0]
    # comments = ['xview_syn_xview_bkg_px23whr3_rnd_bwratio_models_color','xview_syn_xview_bkg_px23whr3_rnd_bwratio_models_mixed']
    # syn_ratios = [1, 1]
    # comments = ['xview_syn_xview_bkg_px23whr3_rnd_bwratio_models_color','xview_syn_xview_bkg_px23whr3_rnd_bwratio_models_mixed',
    #             'syn_xview_bkg_px23whr3_rnd_bwratio_models_color', 'syn_xview_bkg_px23whr3_rnd_bwratio_models_mixed']
    # syn_ratios = [1, 1, None, None]
    # comments = ['syn_xview_bkg_px23whr3_rnd_bwratio_models_mixed']
    # syn_ratios = [None]
    # comments = ['xview_syn_xview_bkg_px23whr3_rnd_bwratio_flat0.8_models_color','xview_syn_xview_bkg_px23whr3_rnd_bwratio_flat0.8_models_mixed',
    #             'syn_xview_bkg_px23whr3_rnd_bwratio_flat0.8_models_color', 'syn_xview_bkg_px23whr3_rnd_bwratio_flat0.8_models_mixed']
    # syn_ratios = [1, 1, None, None]
    # comments = ['px23whr3', 'xview_syn_xview_bkg_px23whr3_6groups_models_color', 'xview_syn_xview_bkg_px23whr3_6groups_models_mixed']
    # syn_ratios = [0, 1, 1]
    # comments = ['xview_syn_xview_bkg_px23whr3_rnd_bwratio_asx_models_color', 'xview_syn_xview_bkg_px23whr3_rnd_bwratio_asx_models_mixed',
    #             'syn_xview_bkg_px23whr3_rnd_bwratio_asx_models_color', 'syn_xview_bkg_px23whr3_rnd_bwratio_asx_models_mixed']
    # syn_ratios = [1, 1, None, None]
    # comments = ['xview_syn_xview_bkg_px23whr3_xratio_xcolor_models_color', 'xview_syn_xview_bkg_px23whr3_xratio_xcolor_models_mixed',
    #             'syn_xview_bkg_px23whr3_xratio_xcolor_models_color', 'syn_xview_bkg_px23whr3_xratio_xcolor_models_mixed']
    # syn_ratios = [1, 1, None, None]
    # comments = ['xview_syn_xview_bkg_px23whr3_sbwratio_xratio_xcolor_models_color', 'xview_syn_xview_bkg_px23whr3_sbwratio_xratio_xcolor_models_mixed',
    #             'syn_xview_bkg_px23whr3_sbwratio_xratio_xcolor_models_color', 'syn_xview_bkg_px23whr3_sbwratio_xratio_xcolor_models_mixed']
    # syn_ratios = [1, 1, None, None]
    # comments = ['syn_xview_bkg_px23whr3_sbwratio_xratio_xcolor_models_color', 'syn_xview_bkg_px23whr3_sbwratio_xratio_xcolor_models_mixed']
    # syn_ratios = [None, None]
    # comments = ['xview_syn_xview_bkg_px23whr3_sbwratio_xratio_xcolor_dark_models_color', 'xview_syn_xview_bkg_px23whr3_sbwratio_xratio_xcolor_dark_models_mixed']
    # syn_ratios = [1, 1]
    # comments = ['xview_syn_xview_bkg_px23whr3_sbwratio_new_xratio_xcolor_models_color', 'xview_syn_xview_bkg_px23whr3_sbwratio_new_xratio_xcolor_models_mixed']
    # syn_ratios = [1, 1]
    # comments = ['syn_xview_bkg_px23whr3_sbwratio_xratio_xcolor_dark_models_color', 'syn_xview_bkg_px23whr3_sbwratio_xratio_xcolor_dark_models_mixed']
    # syn_ratios = [None, None]
    # comments = ['xview_syn_xview_bkg_px23whr3_xbw_xcolor_model0_color', 'xview_syn_xview_bkg_px23whr3_xbw_xcolor_model0_mixed',
    #             'syn_xview_bkg_px23whr3_xbw_xcolor_model0_color', 'syn_xview_bkg_px23whr3_xbw_xcolor_model0_mixed']
    # syn_ratios = [1, 1, None, None]
    # comments = ['syn_xview_bkg_px23whr3_xbw_xcolor_model0_color', 'syn_xview_bkg_px23whr3_xbw_xcolor_model0_mixed']
    # syn_ratios = [None, None]
    # comments = ['xview_syn_xview_bkg_px23whr3_xbw_xrxc_spr_sml_models_color', 'xview_syn_xview_bkg_px23whr3_xbw_xrxc_spr_sml_models_mixed',
    #             'px23whr3']
    # syn_ratios = [1, 1, 0]#,
    # comments = ['syn_xview_bkg_px23whr3_xbw_xrxc_spr_sml_models_color', 'syn_xview_bkg_px23whr3_xbw_xrxc_spr_sml_models_mixed']
    # syn_ratios = [ None, None]
    # comments = ['px23whr3']
    # syn_ratios = [0]
    # comments = ['xview_syn_xview_bkg_px15whr3_xbw_xcolor_model4_color', 'xview_syn_xview_bkg_px15whr3_xbw_xcolor_model4_mixed',
    #             'xview_syn_xview_bkg_px23whr3_xbw_xcolor_model0_color', 'xview_syn_xview_bkg_px23whr3_xbw_xcolor_model0_mixed']
    # syn_ratios = [1, 1, 1, 1]
    # comments = ['xview_syn_xview_bkg_px15whr3_xbw_xcolor_model4_color', 'xview_syn_xview_bkg_px15whr3_xbw_xcolor_model4_mixed']
    # syn_ratios = [1, 1]
    # comments = ['xview_syn_xview_bkg_px23whr3_xbw_xcolor_model1_color', 'xview_syn_xview_bkg_px23whr3_xbw_xcolor_model1_mixed']
    # syn_ratios = [1, 1]
    # comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_model4_v1_color', 'syn_xview_bkg_px15whr3_xbw_xcolor_model4_v1_mixed']
    # comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_model4_v2_color', 'syn_xview_bkg_px15whr3_xbw_xcolor_model4_v2_mixed']
    # comments = ['syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_gauss_model4_v3_color', 'syn_xview_bkg_px15whr3_xbw_xcolor_xbkg_gauss_model4_v3_mixed']
    # syn_ratios = [None, None]
    # comments = ['syn_xview_bkg_px23whr3_xbw_xrxc_spr_sml_models_gauss_color', 'syn_xview_bkg_px23whr3_xbw_xrxc_spr_sml_models_gauss_mixed']
    # comments = ['syn_xview_bkg_px23whr3_xbw_xrxc_model1_gauss_color', 'syn_xview_bkg_px23whr3_xbw_xrxc_model1_gauss_mixed']
    # syn_ratios = [None, None]
    # comments = ['syn_xview_bkg_px23whr3_sbw_xcolor_xbkg_unif_model1_v3_color', 'syn_xview_bkg_px23whr3_sbw_xcolor_xbkg_unif_model1_v3_mixed']
    # syn_ratios = [None, None]
    # comments = ['syn_xview_bkg_px23whr3_xbsw_xcolor_xbkg_gauss_model1_v4_color', 'syn_xview_bkg_px23whr3_xbsw_xcolor_xbkg_gauss_model1_v4_mixed']
    # syn_ratios = [None, None]
    # hyp_cmt = 'hgiou1_fitness'
    # hyp_cmt = 'hgiou1_mean_best'
    # hyp_cmt = 'hgiou1_2gpus'
    # hyp_cmt = 'hgiou1_1gpu_nohsv'
# "syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_scatter_gauss_40_bias0_model5_v1_color",
#     "syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_scatter_gauss_40_color_bias25.5_model5_v2_color",
#     "syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_scatter_gauss_40_color_bias51.0_model5_v3_color",
#     "syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_scatter_gauss_40_color_bias76.5_model5_v4_color",
#     "syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_scatter_gauss_40_color_bias102.0_model5_v5_color",
#     "syn_xview_bkg_px23whr3_xbw_xcolor_xbkg_unif_shdw_scatter_gauss_40_color_bias127.5_model5_v6_color", -----need resume
    opt = get_opt()
    Configure_file = opt.cfg_dict
    cfg_dict = json.load(open(Configure_file))
#    cfg_dict = parse_data_cfg(Configure_file)
    opt.device = cfg_dict['device']
    opt.seed = cfg_dict['seed']
    opt.epochs = cfg_dict['epochs']
    opt.batch_size = cfg_dict['batch_size']
    opt.image_size = cfg_dict['image_size']
    opt.class_num = cfg_dict['class_num']
    opt.cfg = opt.cfg.format(opt.class_num)
    opt.model_id = cfg_dict['model_id']
    opt.conf_thres = cfg_dict['conf_thres']
    opt.nms_iou_thres = cfg_dict['nms_iou_thres']

    comment = cfg_dict['comment']
    base_bias = cfg_dict['base_bias']
    pros = cfg_dict['pros']
    version_base = cfg_dict['version_base']
    prefix = cfg_dict['prefix']

    pxwhrsd = cfg_dict['pxwhrsd']
    hyp_cmt = cfg_dict['hyp_cmt']
    val_syn = cfg_dict['val_syn']
    val_labeled = cfg_dict['val_labeled']
    val_miss = cfg_dict['val_miss']
    # syn_ratios = cfg_dict['syn_ratios']
    hyp = cfg_dict['hyp']
    if 'pw' not in opt.arc:  # remove BCELoss positive weights
        hyp['cls_pw'] = 1.
        hyp['obj_pw'] = 1.
    for cx, pro in enumerate(pros):
        cmt = comment.format(base_bias*pro, version_base+cx) 
        cinx = cmt.find('_RC') # first letter index
        endstr = cmt[cinx:]
        rcinx = endstr.rfind('_')
        sstr = endstr[:rcinx]
        if cinx >= 0:
            suffix = sstr
        else:
            suffix = ''

        opt.name = prefix + suffix

        opt.base_dir = opt.base_dir.format(opt.class_num, pxwhrsd.format(opt.seed))
        if val_syn:
            hyp_cmt_name = hyp_cmt + '_val_syn'
            opt.model_id = None
            opt.data = 'data_xview/{}_{}_cls/{}_seed{}/{}_seed{}.data'.format(cmt, opt.class_num, cmt, opt.seed, cmt, opt.seed)
        elif val_labeled:
            hyp_cmt_name = hyp_cmt + '_val_labeled'
            opt.data = 'data_xview/{}_{}_cls/{}_seed{}/{}_seed{}_xview_val_labeled.data'.format(cmt, opt.class_num, cmt, opt.seed, cmt, opt.seed)
        elif val_miss:
            hyp_cmt_name = hyp_cmt + '_val_labeled_miss'
            opt.data = 'data_xview/{}_{}_cls/{}_seed{}/{}_seed{}_xview_val_labeled_miss.data'.format(cmt, opt.class_num, cmt, opt.seed, cmt, opt.seed)
        elif opt.model_id < 0:
            hyp_cmt_name = hyp_cmt + 'xview_only'
            opt.data = 'data_xview/{}_cls/{}_seed{}/xview_{}_seed{}.data'.format(opt.class_num, cmt, opt.seed, cmt, opt.seed)
        else:
            hyp_cmt_name = hyp_cmt + '_val_xview'
            opt.data = 'data_xview/{}_{}_cls/{}_seed{}/{}_seed{}_xview_val.data'.format(cmt, opt.class_num, cmt, opt.seed, cmt, opt.seed)

        time_marker = time.strftime('%Y-%m-%d_%H.%M', time.localtime())
#        time_marker = '2020-06-30_10.03'
        opt.weights_dir = 'weights/{}_cls/{}_seed{}/{}/'.format(opt.class_num, cmt, opt.seed, '{}_{}_seed{}'.format(time_marker, hyp_cmt_name, opt.seed))
        opt.writer_dir = 'writer_output/{}_cls/{}_seed{}/{}/'.format(opt.class_num, cmt, opt.seed, '{}_{}_seed{}'.format(time_marker, hyp_cmt_name, opt.seed))
        opt.result_dir = 'result_output/{}_cls/{}_seed{}/{}/'.format(opt.class_num, cmt, opt.seed, '{}_{}_seed{}'.format(time_marker, hyp_cmt_name, opt.seed))

        if not os.path.exists(opt.weights_dir):
            os.makedirs(opt.weights_dir)

        if not os.path.exists(opt.writer_dir):
            os.makedirs(opt.writer_dir)

        if not os.path.exists(opt.result_dir):
            os.makedirs(opt.result_dir)
        results_file = os.path.join(opt.result_dir, 'results_{}_seed{}.txt'.format(opt.name, opt.seed))
        last = os.path.join(opt.weights_dir, 'last_seed{}.pt'.format(opt.seed))
        best = os.path.join(opt.weights_dir, 'best_seed{}.pt'.format(opt.seed))
        opt.weights = last if opt.resume else opt.weights
        print(opt)
        # scale hyp['obj'] by img_size (evolved at 320)
        # hyp['obj'] *= opt.img_size / 320.

        if not opt.evolve:  # Train normally
            # prebias()  # optional
            train(opt)  # train normally
            # plot_results(result_dir=opt.result_dir, png_name='results_{}_{}.png'.format(opt.syn_display_type, opt.syn_ratio))
        else:  # Evolve hyperparameters (optional)
            opt.notest = True  # only test final epoch
            opt.nosave = True  # only save final checkpoint
            if opt.bucket:
                os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

            for _ in range(1):  # generations to evolve
                if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                    # Select parent(s)
                    x = np.loadtxt('evolve.txt', ndmin=2)
                    parent = 'weighted'  # parent selection method: 'single' or 'weighted'
                    if parent == 'single' or len(x) == 1:
                        x = x[fitness(x).argmax()]
                    elif parent == 'weighted':  # weighted combination
                        n = min(10, x.shape[0])  # number to merge
                        x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                        w = fitness(x) - fitness(x).min()  # weights
                        x = (x[:n] * w.reshape(n, 1)).sum(0) / w.sum()  # new parent
                    for i, k in enumerate(hyp.keys()):
                        hyp[k] = x[i + 7]

                    # Mutate
                    np.random.seed(int(time.time()))
                    s = np.random.random() * 0.15  # sigma
                    g = [1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # gains
                    for i, k in enumerate(hyp.keys()):
                        x = (np.random.randn() * s * g[i] + 1) ** 2.0  # plt.hist(x.ravel(), 300)
                        hyp[k] *= float(x)  # vary by sigmas

                # Clip to limits
                keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
                limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
                for k, v in zip(keys, limits):
                    hyp[k] = np.clip(hyp[k], v[0], v[1])

                # Train mutation
                # prebias()
                results = train()

                # Write mutation results
                print_mutation(hyp, results, opt.bucket)


