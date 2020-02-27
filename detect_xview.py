import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets_xview import *
from utils.utils_xview import *
from utils.torch_utils import *


def detect(save_img=False):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    # out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    # webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    out, weights, half, view_img, save_txt = opt.output, opt.weights, opt.half, opt.view_img, opt.save_txt
    data = opt.data
    # Configure run
    data_dict = parse_data_cfg(data)
    source = data_dict['valid']
    # test_label_path = data_dict['valid_label']
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    # attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=False, opset_version=10)

        # Validate exported model
        import onnx
        model = onnx.load('weights/export.onnx')  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=half)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)

    # Get names and colors
    #fixme
    # names = load_classes(opt.names)
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    df_cat_color = pd.read_csv(os.path.join(opt.data_dir, 'categories_id_color_diverse_{}.txt'.format(opt.class_num)), delimiter='\t')
    names = df_cat_color['category'].tolist()
    colors = df_cat_color['color'].tolist()

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, time.time() - t))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-{}cls_syn.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data_xview/{}_cls/xview.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/{}_cls/{}_{}/best_{}_{}.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='data_xview/{}_cls/', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='runs/{}_cls/{}_{}/', help='output folder')  # output folder
    parser.add_argument('--data_dir', type=str, default='data_xview/{}_cls/', help='data xview folder')  # output folder
    parser.add_argument('--data', type=str, default='data_xview/{}_cls/xview_{}_{}.data', help='data xview folder')
    parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')

    parser.add_argument('--class_num', type=int, default=1, help='class number') # 60
    parser.add_argument("--syn_ratio", type=float, default=0.25, help="ratio of synthetic data: 0 0.25, 0.5, 0.75, 1.0")
    parser.add_argument('--syn_display_type', type=str, default='syn_texture', help='syn, syn_texture, syn_color')

    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    opt = parser.parse_args()
    opt.cfg = opt.cfg.format(opt.class_num)
    opt.names = opt.names.format(opt.class_num)
    opt.source = opt.source.format(opt.class_num)
    opt.weights = opt.weights.format(opt.class_num, opt.syn_display_type, opt.syn_ratio, opt.syn_display_type, opt.syn_ratio)
    opt.data = opt.data.format(opt.class_num, opt.syn_display_type, opt.syn_ratio)
    opt.data_dir = opt.data_dir.format(opt.class_num)
    opt.output = opt.output .format(opt.class_num, opt.syn_display_type, opt.syn_ratio)
    print(opt)

    with torch.no_grad():
        detect()
