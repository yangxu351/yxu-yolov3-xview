import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import test_xview as test  # import test.py to get mAP after each epoch
from models_xview import *
from utils import torch_utils
from utils.datasets_xview import *
# from utils.datasets_xview_backup import *
from utils.utils_xview import *
from utils.torch_utils import *
import warnings
warnings.filterwarnings("ignore")

GPU="0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed

# Hyperparameters (results68: 59.2 mAP@0.5 yolov3-spp-416) https://github.com/ultralytics/yolov3/issues/310

hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 49.5,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0': 0.00579,  # initial learning rate (SGD=1E-3, Adam=9E-5)
       'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.5,  # focal loss gamma
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98,  # image rotation (+/- deg)
       'translate': 0.05,  # image translation (+/- fraction)
       'scale': 0.05,  # image scale (+/- gain)
       'shear': 0.641}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v


def infi(dl, loop_count):
    while True:
        num_indices = np.random.permutation(len(dl))
        nb = len(dl) // opt.batch_size
        j = 0
        for i in range(loop_count):
            if j == nb:
                j = 0
            index_batch = num_indices[j*opt.batch_size: (j+1)*opt.batch_size]
            j += 1
            yield dl[index_batch]


def infi_loop(dl):
    while True:
        for (imgs, targets, paths, _) in dl:
            yield imgs, targets, paths


def train():
    cfg = opt.cfg
    data = opt.data
    img_size = opt.img_size
    epochs = 1 if opt.prebias else opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt.weights  # initial training weights

    if 'pw' not in opt.arc:  # remove BCELoss positive weights
        hyp['cls_pw'] = 1.
        hyp['obj_pw'] = 1.

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
    loop_count = int(syn_0_xview_number)//batch_size
    # Remove previous results
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
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
    best_fitness = float('inf')
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

    # Scheduler https://github.com/ultralytics/yolov3/issues/238
    # lf = lambda x: 1 - x / epochs  # linear ramp to zero
    # lf = lambda x: 10 ** (hyp['lrf'] * x / epochs)  # exp ramp
    # lf = lambda x: 1 - 10 ** (hyp['lrf'] * (1 - x / epochs))  # inverse exp ramp
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=range(59, 70, 1), gamma=0.8)  # gradual fall to 0.1*lr0
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    # # Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        #fixme
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataset
    dataset = LoadImagesAndLabels(train_path, train_label_path, img_size, batch_size,
                                  class_num=opt.class_num,
                                  augment=True, # False, #True,
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
        testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, test_label_path, opt.img_size, batch_size * 2, class_num=opt.class_num,
                                                                     hyp=hyp,
                                                                     rect=True,
                                                                     cache_labels=True,
                                                                     cache_images=opt.cache_images),
                                                 batch_size=batch_size * 2,
                                                 num_workers=nw,
                                                 pin_memory=True,
                                                 collate_fn=dataset.collate_fn)

    # Start training
    #fixme
    # nb = len(dataloader)
    nb = loop_count
    print('nb ', nb)
    model.nc = nc  # attach number of classes to model
    model.arc = opt.arc  # attach yolo architecture
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    maps = np.zeros(nc)  # mAP per class
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    torch_utils.model_info(model, report='summary')  # 'full' or 'summary'
    print('Using %g dataloader workers' % nw)
    print('Starting %s for %g epochs...' % ('prebias' if opt.prebias else 'training', epochs))
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
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

        gen_data = infi_loop(dataloader)
        for i in range(nb):
            imgs, targets, paths = next(gen_data)
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            # Multi-Scale training
            if opt.multi_scale:
                if ni / accumulate % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                    img_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Plot images with bounding boxes
            if ni == 0:
                fname = 'train_batch%g.jpg' % i
                plot_images(imgs=imgs, targets=targets, paths=paths, fname=fname)
                if tb_writer:
                    tb_writer.add_image(fname, cv2.imread(fname)[:, :, ::-1], dataformats='HWC')

            # Hyperparameter burn-in
            # n_burn = nb - 1  # min(nb // 5 + 1, 1000)  # number of burn-in batches
            # if ni <= n_burn:
            #     for m in model.named_modules():
            #         if m[0].endswith('BatchNorm2d'):
            #             m[1].momentum = 1 - i / n_burn * 0.99  # BatchNorm2d momentum falls from 1 - 0.01
            #     g = (i / n_burn) ** 4  # gain rises from 0 - 1
            #     for x in optimizer.param_groups:
            #         x['lr'] = hyp['lr0'] * g
            #         x['weight_decay'] = hyp['weight_decay'] * g

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

        # Process epoch results
        final_epoch = epoch + 1 == epochs
        if opt.prebias:
            print_model_biases(model)
        elif not opt.notest or final_epoch:  # Calculate mAP
            #fixme
            is_xview = any([x in data for x in ['xview_{}_{}.data'.format(opt.syn_display_type, opt.syn_ratio)]]) and model.nc == opt.class_num
            results, maps = test.test(cfg,
                                      data,
                                      batch_size=batch_size * 2,
                                      img_size=opt.img_size, 
                                      conf_thres=0.001 if final_epoch else 0.1,  # 0.1 for speed
                                      iou_thres=0.6 if final_epoch and is_xview else 0.5,
                                      save_json=True, # final_epoch and is_xview, #fixme
                                      model=model,
                                      dataloader=testloader, opt=opt)

        # Write epoch results
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        if len(opt.name) and opt.bucket and not opt.prebias:
            os.system('gsutil cp results.txt gs://%s/results%s.txt' % (opt.bucket, opt.name))

        # Write Tensorboard results
        if tb_writer:
            x = list(mloss) + list(results)
            titles = ['GIoU', 'Objectness', 'Classification', 'Train loss',
                      'Precision', 'Recall', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification']
            for xi, title in zip(x, titles):
                tb_writer.add_scalar(title, xi, epoch)

        # Update best mAP
        fitness = sum(results[4:])  # total loss
        if fitness < best_fitness:
            best_fitness = fitness

        # Save training results
        save = (not opt.nosave) or (final_epoch and not opt.evolve) or opt.prebias
        if save:
            with open(results_file, 'r') as f:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': f.read(),
                         'model': model.module.state_dict() if type(
                             model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                         'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last checkpoint
            torch.save(chkpt, last)

            # Save best checkpoint
            if best_fitness == fitness:
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, opt.weights_dir + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt

        # end epoch ----------------------------------------------------------------------------------------------------

    # end training
    # if len(opt.name) and not opt.prebias:
    #     fresults, flast, fbest = 'results%s.txt' % opt.name, 'last%s.pt' % opt.name, 'best%s.pt' % opt.name
    #     os.rename(opt.result_dir + 'results.txt', fresults)
    #     os.rename(opt.weights_dir + 'last.pt', opt.weights_dir + flast) if os.path.exists(opt.weights_dir + 'last.pt') else None
    #     os.rename(opt.weights_dir + 'best.pt', opt.weights_dir + fbest) if os.path.exists(opt.weights_dir + 'best.pt') else None
    #
    #     # save to cloud
    #     if opt.bucket:
    #         os.system('gsutil cp %s %s gs://%s' % (fresults, opt.weights_dir + flast, opt.bucket))

    plot_results(result_dir=opt.result_dir, png_name='results_{}_{}.png'.format(opt.syn_display_type, opt.syn_ratio), class_num=opt.class_num)  # save as results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()

    return results


def prebias():
    # trains output bias layers for 1 epoch and creates new backbone
    if opt.prebias:
        # opt_0 = opt  # save settings
        # opt.rect = False  # update settings (if any)

        train()  # train model biases
        create_backbone(last)  # saved results as backbone.pt

        # opt = opt_0  # reset settings
        opt.weights = opt.weights_dir + 'backbone.pt'  # assign backbone
        opt.prebias = False  # disable prebias


def get_opt(dt, sr=None, comments=''):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=180)  # 500200 batches at bs 16, 117263 images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=8)  # effective bs = batch_size * accumulate = 16 * 4 = 64

    parser.add_argument('--accumulate', type=int, default=4, help='batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-{}cls_syn.cfg', help='*.cfg path')
    if sr == 0:
        parser.add_argument('--data', type=str, default='data_xview/{}_cls/{}/xview_{}_{}{}.data', help='*.data path')
    elif comments == '_syn_only':
        parser.add_argument('--data', type=str, default='data_xview/{}_cls/{}_syn_only.data', help='*.data path')
    else:
        parser.add_argument('--data', type=str, default='data_xview/{}_cls/xview_{}_{}/xview_{}_{}{}.data', help='*.data path')
    parser.add_argument('--writer_dir', type=str, default='writer_output/{}_cls/{}_{}/', help='*events* path')
    parser.add_argument('--weights_dir', type=str, default='weights/{}_cls/{}_{}/', help='to save weights path')
    parser.add_argument('--result_dir', type=str, default='result_output/{}_cls/{}_{}/', help='to save result files path')

    parser.add_argument("--syn_ratio", type=float, default=sr, help="ratio of synthetic data: 0.25, 0.5, 0.75,  0")
    parser.add_argument('--syn_display_type', type=str, default=dt, help='syn_texture0, syn_color0, syn_texture, syn_color, syn_mixed, syn (match 0)')

    parser.add_argument('--multi_scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--img_size', type=int, default=608, help='inference size (pixels)') # 416 608
    parser.add_argument('--class_num', type=int, default=1, help='class number') # 60 6 1

    parser.add_argument('--rect', default=False, action='store_true', help='rectangular training')
    parser.add_argument('--resume', default=False,  action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache_images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='', help='initial weights') # weights/ultralytics68.pt
    parser.add_argument('--arc', type=str, default='default', help='yolo architecture')  # defaultpw, uCE, uBCE
    parser.add_argument('--prebias', action='store_true', help='pretrain model biases')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--var', type=float, help='debug variable')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    # display_type = ['syn_texture', 'syn_color', 'syn_mixed']
    # syn_ratio = [0.25, 0.5, 0.75]
    # display_type = ['syn'] #'syn_mixed',
    # syn_ratio = [0]  # 0.75, 0.5,
    # display_type = ['syn_color'] #'syn_mixed',

    # syn_ratio = [0.25, 0.5, 0.75]
    # trial = 3
    # mis_ratio = [0.025, 0.05]
    # for dt in display_type:
    #     for sr in syn_ratio:
    #         for mr in mis_ratio:
    #             for i in range(trial):

    # display_type = ['syn_texture0'] # , 'syn_color0']
    # syn_ratio = [0.75]

    display_type = ['syn_background']
    # syn_ratio = [0.3, 0.2, 0.1] #
    syn_ratio = [0]
    #fixme
    # comments = ''
    # comments = '_38bbox_000wh'
    # comments = '_38bbox_giou0'
    # comments = '_mismatch{}_{}'.format(mr, i)
    # comments = '_with_model'
    # comments = '_px4whr3'
    # comments = '_syn_only'
    # comments = '_px6whr4_giou0'
    # comments = '_px6whr4_rect_mosaic'
    # comments = '_px6whr4_giou0_seed17'
    # cmts = '_px6whr4_ng0_seed17'
    comments = '_px6whr4_ng0'
    cmts = '_px6whr4_ng0'
    for dt in display_type:
        for sr in syn_ratio:
            opt = get_opt(dt, sr, comments)
            # opt = get_opt(dt, comments=comments)
            opt.cfg = opt.cfg.format(opt.class_num)
            time_marker = time.strftime('%Y-%m-%d_%H.%M', time.localtime())
            # time_marker = '2020-03-25_08.12'
            opt.weights_dir = opt.weights_dir.format(opt.class_num, opt.syn_display_type, opt.syn_ratio) + '{}/'.format(time_marker + cmts)
            opt.writer_dir = opt.writer_dir.format(opt.class_num, opt.syn_display_type, opt.syn_ratio) + '{}/'.format(time_marker + cmts)
            if comments == '_syn_only':
                opt.data = opt.data.format(opt.class_num, opt.syn_display_type)
                opt.result_dir = opt.result_dir.format(opt.class_num, opt.syn_display_type, comments) + '{}/'.format(time_marker + cmts)
                results_file = os.path.join(opt.result_dir, 'results_{}_{}.txt'.format(opt.syn_display_type, comments))
                last = os.path.join(opt.weights_dir, 'last_{}_{}.pt'.format(opt.syn_display_type, comments))
                best = os.path.join(opt.weights_dir, 'best_{}_{}.pt'.format(opt.syn_display_type, comments))
            else:
                if sr == 0 and not comments:
                    opt.data = opt.data.format(opt.class_num, opt.syn_display_type, opt.syn_ratio, comments)
                if sr == 0 and comments:
                    opt.data = opt.data.format(opt.class_num, comments[1:], opt.syn_display_type, opt.syn_ratio, comments)
                else:
                    opt.data = opt.data.format(opt.class_num, opt.syn_display_type, opt.syn_ratio, opt.syn_display_type, opt.syn_ratio, comments)
                opt.result_dir = opt.result_dir.format(opt.class_num, opt.syn_display_type, opt.syn_ratio) + '{}/'.format(time_marker + cmts)
                results_file = os.path.join(opt.result_dir, 'results_{}_{}.txt'.format(opt.syn_display_type, opt.syn_ratio))
                last = os.path.join(opt.weights_dir, 'last_{}_{}.pt'.format(opt.syn_display_type, opt.syn_ratio))
                best = os.path.join(opt.weights_dir, 'best_{}_{}.pt'.format(opt.syn_display_type, opt.syn_ratio))
            # opt.name = '_{}_{}'.format(opt.syn_display_type, opt.syn_ratio)
            if not os.path.exists(opt.weights_dir):
                os.makedirs(opt.weights_dir)

            if not os.path.exists(opt.writer_dir):
                os.makedirs(opt.writer_dir)

            if not os.path.exists(opt.result_dir):
                os.makedirs(opt.result_dir)

            opt.weights = last if opt.resume else opt.weights
            print(opt)
            device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
            if device.type == 'cpu':
                mixed_precision = False

            # scale hyp['obj'] by img_size (evolved at 320)
            # hyp['obj'] *= opt.img_size / 320.

            tb_writer = None
            if not opt.evolve:  # Train normally
                try:
                    # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
                    from torch.utils.tensorboard import SummaryWriter
                    tb_writer = SummaryWriter(log_dir=opt.writer_dir)
                except:
                    pass
                prebias()  # optional
                train()  # train normally
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
                    prebias()
                    results = train()

                    # Write mutation results
                    print_mutation(hyp, results, opt.bucket)

                    # Plot results
                    # plot_evolution_results(hyp)

    '''
     ######***0
    '''
    # dt = 'syn_background'
    # sr = 0
    # opt = get_opt(dt, sr, comments)
    # # opt = get_opt(dt, comments=comments)
    # opt.cfg = opt.cfg.format(opt.class_num)
    # time_marker = time.strftime('%Y-%m-%d_%H.%M', time.localtime())
    # # time_marker = '2020-03-25_08.12'
    # opt.weights_dir = opt.weights_dir.format(opt.class_num, opt.syn_display_type, opt.syn_ratio) + '{}/'.format(time_marker + cmts)
    # opt.writer_dir = opt.writer_dir.format(opt.class_num, opt.syn_display_type, opt.syn_ratio) + '{}/'.format(time_marker + cmts)
    # if comments == '_syn_only':
    #     opt.data = opt.data.format(opt.class_num, opt.syn_display_type)
    #     opt.result_dir = opt.result_dir.format(opt.class_num, opt.syn_display_type, comments) + '{}/'.format(time_marker + comments)
    #     results_file = os.path.join(opt.result_dir, 'results_{}_{}.txt'.format(opt.syn_display_type, comments))
    #     last = os.path.join(opt.weights_dir, 'last_{}_{}.pt'.format(opt.syn_display_type, comments))
    #     best = os.path.join(opt.weights_dir, 'best_{}_{}.pt'.format(opt.syn_display_type, comments))
    # else:
    #     opt.data = opt.data.format(opt.class_num, comments[1:], dt, sr, comments)
    #     opt.result_dir = opt.result_dir.format(opt.class_num, opt.syn_display_type, opt.syn_ratio) + '{}/'.format(time_marker + cmts)
    #     results_file = os.path.join(opt.result_dir, 'results_{}_{}.txt'.format(opt.syn_display_type, opt.syn_ratio))
    #     last = os.path.join(opt.weights_dir, 'last_{}_{}.pt'.format(opt.syn_display_type, opt.syn_ratio))
    #     best = os.path.join(opt.weights_dir, 'best_{}_{}.pt'.format(opt.syn_display_type, opt.syn_ratio))
    # # opt.name = '_{}_{}'.format(opt.syn_display_type, opt.syn_ratio)
    # if not os.path.exists(opt.weights_dir):
    #     os.makedirs(opt.weights_dir)
    #
    # if not os.path.exists(opt.writer_dir):
    #     os.makedirs(opt.writer_dir)
    #
    # if not os.path.exists(opt.result_dir):
    #     os.makedirs(opt.result_dir)
    #
    # opt.weights = last if opt.resume else opt.weights
    # print(opt)
    # device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    # if device.type == 'cpu':
    #     mixed_precision = False
    #
    # # scale hyp['obj'] by img_size (evolved at 320)
    # # hyp['obj'] *= opt.img_size / 320.
    #
    # tb_writer = None
    # if not opt.evolve:  # Train normally
    #     try:
    #         # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
    #         from torch.utils.tensorboard import SummaryWriter
    #         tb_writer = SummaryWriter(log_dir=opt.writer_dir)
    #     except:
    #         pass
    #     prebias()  # optional
    #     train()  # train normally
    #     # plot_results(result_dir=opt.result_dir, png_name='results_{}_{}.png'.format(opt.syn_display_type, opt.syn_ratio))
    # else:  # Evolve hyperparameters (optional)
    #     opt.notest = True  # only test final epoch
    #     opt.nosave = True  # only save final checkpoint
    #     if opt.bucket:
    #         os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists
    #
    #     for _ in range(1):  # generations to evolve
    #         if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
    #             # Select parent(s)
    #             x = np.loadtxt('evolve.txt', ndmin=2)
    #             parent = 'weighted'  # parent selection method: 'single' or 'weighted'
    #             if parent == 'single' or len(x) == 1:
    #                 x = x[fitness(x).argmax()]
    #             elif parent == 'weighted':  # weighted combination
    #                 n = min(10, x.shape[0])  # number to merge
    #                 x = x[np.argsort(-fitness(x))][:n]  # top n mutations
    #                 w = fitness(x) - fitness(x).min()  # weights
    #                 x = (x[:n] * w.reshape(n, 1)).sum(0) / w.sum()  # new parent
    #             for i, k in enumerate(hyp.keys()):
    #                 hyp[k] = x[i + 7]
    #
    #             # Mutate
    #             np.random.seed(int(time.time()))
    #             s = np.random.random() * 0.15  # sigma
    #             g = [1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # gains
    #             for i, k in enumerate(hyp.keys()):
    #                 x = (np.random.randn() * s * g[i] + 1) ** 2.0  # plt.hist(x.ravel(), 300)
    #                 hyp[k] *= float(x)  # vary by sigmas
    #
    #         # Clip to limits
    #         keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
    #         limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
    #         for k, v in zip(keys, limits):
    #             hyp[k] = np.clip(hyp[k], v[0], v[1])
    #
    #         # Train mutation
    #         prebias()
    #         results = train()
    #
    #         # Write mutation results
    #         print_mutation(hyp, results, opt.bucket)

            # Plot results
            # plot_evolution_results(hyp)
