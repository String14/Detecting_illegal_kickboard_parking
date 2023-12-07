# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

from PIL import Image, ImageOps
import os

import cv2
import os
import time
import tensorflow as tf
from model.pspunet import pspunet
from data_loader.display import create_mask
import numpy as np

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
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

    #################
    # 이미지 리사이징 #
    #################

    raw_path = (source + '/' if source[len(source)-1] != '/' else source) # 원본 이미지 경로
    token_list = os.listdir(raw_path)  # 원본 이미지 경로 내 폴더들 list
    data_path = 'data/640_size_data/'  # 저장할 이미지 경로

    for token in token_list:
        # 원본 이미지 경로와 저장할 경로 이미지 지정
        image_path = raw_path + token
        save_path = data_path + token

        # 이미지 열기
        im = Image.open(image_path)

        # 이미지 resize
        im = im.resize((640, 640))
        im = ImageOps.exif_transpose(im)
        # im.show()

        # 이미지 JPG로 저장
        im = im.convert('RGB')
        im.save(save_path, 'JPEG')

    source = data_path

    #################
    #################

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
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

        # Define the path for the CSV file
        csv_path = save_dir / 'predictions.csv'

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

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
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

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
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

    # gpu 사용에 대한 설정
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=500)])
        except RuntimeError as e:
            print(e)

    # 킥보드 인식 결과 파일 읽기
    file_path = "runs/detect"
    file_list = os.listdir(file_path)
    IMG_saveOption = True

    path = 'data/640_size_data/'
    token_list = os.listdir(path)  # 원본 이미지 경로 내 폴더들 list

    filePath = []
    xywh = []
    # 각 이미지당 킥보드가 검출된 좌표값을 읽어들여 x, y, w, h를 계산하고 저장
    for idx in range(len(token_list)):
        try:
            f = open(file_path + "/exp" + str(len(file_list)) + "/labels/" + token_list[idx][:-4] + ".txt", 'r')

            lines = f.readlines()
            line_xywh = [[] for line_num in range(len(lines))]
            # 한 이미지에 여러개의 킥보드 객체가 있을 경우를 대비
            for line_num in range(len(lines)):
                line = lines[line_num][:-1].split(" ")
                cls = line[0]
                x = round(float(line[1]) * 640)
                y = round(float(line[2]) * 640)
                w = round(float(line[3]) * 640)
                h = round(float(line[4]) * 640)

                line_xywh[line_num] = [x, y, w, h]

            # xywh [n번째 이미지의] [n번째 킥보드 객체] [의 x, y, w, h 값]
            xywh.append(line_xywh)
            f.close()

        except:
            # 해당 사진에서 킥보드가 단 한개도 검출되지 않았을 경우
            xywh.append([[None]])

    print(xywh)

    # data 경로 설정
    # path = './image/'
    # fileName = 'test03'
    # IMG_saveOption = True
    # filePath = os.path.join(path, fileName + ('.jpg' if IMG_saveOption else '.mp4'))
    # if os.path.isfile(filePath):  # 해당 파일이 있는지 확인
    #     # 영상 객체(파일) 가져오기
    #     cap = cv2.VideoCapture(filePath)
    # else:
    #     print("file not exist")

    filePath = []
    cap = []
    for idx in range(len(token_list)):
        filePath.append(os.path.join(path, token_list[idx][:-4] + ('.jpg' if IMG_saveOption else '.mp4')))

    for idx in range(len(filePath)):
        if os.path.isfile(filePath[idx]):  # 해당 파일이 있는지 확인
            # 이미지 객체(파일)를 cap으로 가져오기
            cap.append(cv2.VideoCapture(filePath[idx]))
        else:
            print(filePath[idx] + " file not exist")

    # 이미지 크기 지정값으로 변환
    IMG_WIDTH = 640
    IMG_HEIGHT = 640
    n_classes = 7

    model = pspunet((IMG_HEIGHT, IMG_WIDTH, 3), n_classes)
    model.load_weights("pspunet_weight.h5")

    for idx in range(len(cap)):
        print("\n이미지 : " + str(idx+1) + "/" + str(len(cap)))
        while True:
            start = time.time()
            try:
                _, frame = cap[idx].read()
                frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
                # frame = cv2.resize(frame, frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 색반전 (BGR -> RGB)
                frame = frame[tf.newaxis, ...]
                frame = frame / 255

            except:
                cv2.destroyAllWindows()
                cap[idx].release()
                break

            pre = model.predict(frame)
            pre = create_mask(pre).numpy()

            # 각 픽셀당 주변상황 정보를 포함한 2차원 배열 출력
            # for i in range(len(pre)): # 640 (y)
            #     for j in range(len(pre[i])): # 640 (x)
            #         print(pre[i][j][0], end=' ')
            #     print()

            # 불법주차 검출 (len(xywh[idx]) -> 이미지 내 킥보드의 갯수, pre [y좌표] [x좌표] [0])
            for kick_num in range(0, len(xywh[idx])):
                if xywh[idx][kick_num] == [None]:
                    print("킥보드를 인식할 수 없습니다. 다시 촬영해주세요.")
                    break

                test_x = xywh[idx][kick_num][0]
                test_y = xywh[idx][kick_num][1]

                test_w = xywh[idx][kick_num][2]
                test_w_half = round(test_w / 2)

                test_h = xywh[idx][kick_num][3]
                test_h_half = round(test_h / 2)

                count_critical_3 = 0
                count_critical_4 = 0

                illegal_parking = False
                # 바운딩 박스의 최하단에서 200px 위쪽까지 확인
                for x_i in range(0, test_w):
                    for y_j in range(200, 0, -1):
                        # 배열은 0부터 시작하므로 -1이 필요하다
                        test_point = pre [test_y + test_h_half - y_j - 1] [test_x - test_w_half + x_i - 1] [0]

                        # 3. 횡단보도 = light green
                        if test_point == 3:
                            count_critical_3 += 1

                        # 4. 점자블록 = yellow
                        elif test_point == 4:
                            count_critical_4 += 1

                        if (count_critical_3 >= 1000) | (count_critical_4 >= 1000):
                            illegal_parking = True
                            print("!!!!!!!!!!!!!!!!!!! 주차 불가 구역 !!!!!!!!!!!!!!!!!!!!")
                            break

                    if illegal_parking: break
                if illegal_parking: break
                else:
                    print("-- 주차 가능 구역 --")
                    break

            frame2 = frame / 2
            # 0. 기타
            # 1. 자전거길 = black
            frame2[0][(pre == 1).all(axis=2)] += [0, 0, 0]  # ""bike_lane_normal", "sidewalk_asphalt", "sidewalk_urethane""
            # 2. 맨홀/나무밑둥 = cyan
            frame2[0][(pre == 2).all(axis=2)] += [0.5, 0.5, 0]  # "caution_zone_stairs", "caution_zone_manhole", "caution_zone_tree_zone", "caution_zone_grating", "caution_zone_repair_zone"]
            # 3. 횡단보도 = light green
            frame2[0][(pre == 3).all(axis=2)] += [0.2, 0.7, 0.5]  # "alley_crosswalk","roadway_crosswalk"
            # 4. 점자블록 = yellow
            frame2[0][(pre == 4).all(axis=2)] += [0, 0.5, 0.5]  # "braille_guide_blocks_normal", "braille_guide_blocks_damaged"
            # 5. 차도 = red
            frame2[0][(pre == 5).all(axis=2)] += [0, 0, 0.5]  # "roadway_normal","alley_normal","alley_speed_bump", "alley_damaged""
            # 6. 보도블럭 = blue
            frame2[0][(pre == 6).all(axis=2)] += [0.5, 0, 0]  # "sidewalk_blocks","sidewalk_cement" , "sidewalk_soil_stone", "sidewalk_damaged","sidewalk_other"

            if IMG_saveOption:
                cv2.imshow('frame', frame2[0])
                cv2.waitKey(2000)
                # TODO : 이미지/영상 저장이 안됨
                # cv2.imwrite('./result/' + token_list[idx] + '.jpg', frame2[0])
            else:
                cv2.imshow('frame', frame2[0])

            video = np.uint8(frame2)
            print(1 / (time.time() - start))
            cv2.waitKey(1)

        cv2.destroyAllWindows()
        cap[idx].release()
