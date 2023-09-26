import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()  # 获取当前文件的绝对路径
ROOT = FILE.parents[0]  # YOLO root directory 获取当前文件所在的根目录
if str(ROOT) not in sys.path:  # 模块的查询路径列表
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative 绝对目录变成相对目录

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    # 读取模型中的一些值，比如模型的步长、模型能检测出来的类别名、模型是不是pytorch的模型类型
    stride, names, pt = model.stride, model.names, model.pt
    # 输入待检查图片的大小，stride就是模型的步长，一般是32，然后会去判断图片的大小（imgsz）是不是32的倍数，如果是的话，那么imgsz就还是原来的尺寸
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt

            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    # 指定网络权重的路径
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp111/weights/best.pt', help='model path or triton URL')
    # 指定要预测的东西输入的路径
    parser.add_argument('--source', type=str, default=ROOT / 'Pictures to be tested', help='file/dir/URL/glob/screen/0(webcam)')
    # 配置文件，里面包含了下载路径和一些数据集基本信息，训练时如果不指定数据集，系统会自己下载coco128数据集。
    parser.add_argument('--data', type=str, default=ROOT / 'data.yaml', help='(optional) dataset.yaml path')
    # 在检测图片前会把图片resize成640 × 640的尺寸，然后再喂进网络里
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    # 置信度的阈值  要大于这个值才会最终有检测框检测出来
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    # 调节IoU的阈值，淘汰（抑制） IOU 大于设定阈值的 BBox，IoU阈值越小越严格，框（检测出来的目标）越少。
    # 多个框的重合度大于IOU阈值的话，置信度较小的就被抑制
    parser.add_argument('--iou-thres', type=float, default=0.4, help='NMS IoU threshold')
    # 大检测数量
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    # 选择GPU用的，如果不指定的话，会自动检测
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # action='store_true'理解成一个“开关”
    # 检测的时候是否实时的把检测结果实时显示出来  终端输入python detect.py --view-img开启
    parser.add_argument('--view-img', action='store_true', help='show results')
    # 是否把检测结果保存成一个.txt的格式  终端输入python detect.py --save-txt  （保存了一些类别信息和边框的位置信息）
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # 是否以.txt的格式保存目标的置信度，单独执行没效果，必须和--save-txt配合使用。  python detect.py --save-txt --save-conf
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # 是否把模型检测的物体裁剪下来
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    # 不保存预测的结果
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # python detect.py --classes 0 只检测类别里面的第一个东西
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    # 跨类别nms 比如待检测图像中有一个长得很像排球的足球，pt文件的分类中有足球和排球两种，那在识别时这个足球可能会被同时框上2个框：一个是足球，一个是排球。
    # 开启agnostic-nms后，那只会框出一个框
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # 这个参数也是一种增强的方式
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # 是否把特征图（feature map）可视化出来
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    # 如果指定这个参数，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息。
    parser.add_argument('--update', action='store_true', help='update all models')
    # 预测结果保存的路径
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    # 预测结果保存的文件夹名字
    parser.add_argument('--name', default='exp', help='save results to project/name')
    # 这个参数的意思就是每次预测模型的结果是否保存在原来的文件夹，如果指定了这个参数的话，
    # 那么本次预测的结果还是保存在上一次保存的文件夹里；如果不指定就是每次预测结果保存一个新的文件夹下。
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # 这个参数就是调节预测框线条粗细的，因为有的时候目标重叠太多会产生遮挡； python detect.py --line-thickness 10
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    # 隐藏标签
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    # 隐藏标签的置信度
    parser.add_argument('--hide-conf', action='store_true', help='hide confidences')
    # 是否使用FP16半精度推理，有利于部署在嵌入式模型里面
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()  # 存储所有的参数信息
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))  # 打印所有参数信息
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))   # 检查一下requirements.txt里面的包有没有成功安装
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()  # 解析参数的函数
    main(opt)
