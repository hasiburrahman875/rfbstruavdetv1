# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
from hashlib import new
import json, pickle
import os
from posixpath import pathsep
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.general import (LOGGER, box_iou, check_dataset, check_img_size, check_requirements, check_suffix, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

aot_results= []
def save_aot_one_pkl(predn, path, file_path=False):
    if not file_path:
        track_id = 0
        for *xyxy, conf, cls in predn.tolist():
            if int(cls) == 0:
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                result = {
                    "detections": [
                        {
                            "track_id": track_id,
                            "x": float(xywh[0]),
                            "y": float(xywh[1]),
                            "w": float(xywh[2]),
                            "h": float(xywh[3]),
                            "n": "airborne",
                            "s": float(conf)
                        }
                    ],
                    "img_name": os.path.basename(path)
                }
                track_id += 1
                aot_results.append(result)
    else:
        os.makedirs(str(Path(file_path).parent), exist_ok=True)
        pickle.dump(aot_results, open(file_path, "wb"))
    
def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b], # XY WH in real coordinates
                      'score': round(p[4], 5)})


jdict_gt = {}
def save_one_json_with_gt(predn, path, class_map, labelsn):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    if len(predn) > 0 or len(labelsn) > 0:
        image_id = int(path.stem) if path.stem.isnumeric() else path.stem
        box = predn[:, :4] #xyxy2xywh(predn[:, :4])  # xywh
        #box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        jdict_gt[image_id] = {"detections":[], "labels":[]}
        for p, b in zip(predn.tolist(), box.tolist()):
            jdict_gt[image_id]["detections"].append({'bbox':[round(x, 3) for x in b], 'score':round(p[4], 5), 'category_id':class_map[int(p[5])]})
        box = labelsn[:, 1:] #xyxy2xywh(labeln[:, 1:])
        #box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner 
        for p, b in zip(labelsn[:, 0], box.tolist()):
            jdict_gt[image_id]["labels"].append({'bbox':[round(x, 3) for x in b], 'category_id':int(p)})
    #return jdict_gt

def save_one_json_with_gt_dump(file_path):
    os.makedirs(str(Path(file_path).parent), exist_ok=True)
    pickle.dump(jdict_gt, open(file_path, "wb"))
    print("Predictions with gt dumped")

def get_data_split_number(data_yaml_file_path):
    split_number = os.path.basename(data_yaml_file_path).split(".")[0].split("_")[-1]
    return int(split_number) if split_number.isnumeric() else 0
    # split_number = int(os.path.basename(data_yaml_file_path).split(".")[0].split("_")[-1].strip()) 
    # return split_number

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

def is_fourth_frame(path):
    return True if int(os.path.basename(path).split(".")[0].split("_")[-1]) % 4 == 0 else False
    

def plot_val_images_for_me(imgs, paths, shapes, targets, predictions, transform_targets=False, main_indices=[]):
    from torchvision.utils import draw_bounding_boxes, save_image
    from torchvision.io import read_image
    from utils.general import extend_iou
    ogimgs = [read_image(str(path)) for path in paths]
    is_any_plotted = False
    #targets[:, 2:] = xywhn2xyxy(targets[:, 2:], w, h)
    for i, img in enumerate(ogimgs):
        if True:#i in main_indices: #
            #target = [target[2:] for target in targets if target[0] == i]
            #target = torch.cat(target, 0).reshape(-1, 4) if len(target) > 0 else torch.zeros((0, 4))
            if len(targets) > 0 :
                #print(f"Targets before scaling {target}, og shape {shapes[i][0]}, ration_pad {shapes[i][1]}, loaded image shape {img.shape[1:]} {imgs[i].shape[1:]}")
                if transform_targets:
                    height, width = imgs[i].shape[1:]
                    target = target*torch.Tensor([width, height, width, height]).to(target.device)
                    target = xywh2xyxy(target)
                    target = scale_coords(imgs[i].shape[1:], target, shapes[i][0], shapes[i][1])
                #print(f"Targets after scaling {target}") 
                img = draw_bounding_boxes(img, targets, width=1, colors="red")
                if len(predictions) > 0:
                    img = draw_bounding_boxes(img, predictions[:, :4], width=1, colors="green")
                save_image(img.float()/255., f"{os.path.basename(paths[i])}_val_check{i}.png")
                print(f"plotted {os.path.basename(paths[i])}")
                is_any_plotted = True
        #break
    
    if is_any_plotted:
        print(paths)
        exit()

@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.5,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        save_json_gt=False,
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        num_frames=5,
        every_fourth_frame=False,
        save_aot_predictions=False,
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None
        ):
    epoch_number = -1
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        check_suffix(weights, '.pt')
        model, epoch_number = attempt_load(weights, map_location=device, return_epoch_number=True)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

        # Data
        data_yaml_path = None
        if isinstance(data, (str, Path)):
            data_yaml_path = data
        if data_yaml_path is None: 
            assert save_aot_predictions == False, print("please launch as main script with '--data AOTTest_1.yaml' param if AOT predictions are to be saved")
        data = check_dataset(data)  # check

    # Half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    model.half() if half else model.float()

    # Configure
    model.eval()
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    #iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()

    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(num_frames, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        annotation_test_path = data[f"annotation_{task}"] if f"annotation_{task}" in data else ""
        video_root = data[f"video_root_path_{task}"] if f"video_root_path_{task}" in data else ""
        dataloader = create_dataloader(data[task], annotation_test_path, video_root, imgsz, batch_size, gs, single_cls, pad=pad, rect=True,
                                       prefix=colorstr(f'{task}: '), is_training=False, num_frames=num_frames)[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    #print(f"names {names}")
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 7) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'epoch('+str(epoch_number)+')')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict = [] 
    stats, ap, ap_class = [], [], []
    #print(f"Is testing every fourth frame ? {every_fourth_frame}, frames {num_frames}, epoch number {epoch_number}, Confidence Threshold {conf_thres}")
    for batch_i, (img, targets, paths, shapes, main_target_indices, label_paths) in enumerate(tqdm(dataloader, desc=s)):
        t1 = time_sync()
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        
        nb, _, height, width = img.shape  # batch size, channels, height, width
        
        t2 = time_sync()
        dt[0] += t2 - t1

        #plot_val_images_for_me(img, paths, shapes, targets, main_target_indices)
        # Run model
        #print(paths)
        out, train_out = model(img, augment=augment)  # inference and training outputs
        
        dt[1] += time_sync() - t2
        
        #exit()
        # Compute loss
        if compute_loss:
            #print(f"val loss computation {targets.shape}")
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls
        
        
        # #select only main frame images ignore other temporal predictions & target
        # out, train_out = out[main_target_indices], [tout[main_target_indices] for tout in train_out]
        # img, paths, shapes, targets = img[main_target_indices], [paths[ii] for ii in main_target_indices], [shapes[jj] for jj in main_target_indices], filter_targets(targets, main_target_indices)
        # label_paths = [label_paths[ii] for ii in main_target_indices]
        
        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        #pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Statistics per image
        for si, pred in enumerate(out):
            if si not in main_target_indices:
                continue
            
            if every_fourth_frame:
                if not is_fourth_frame(paths[si]): #skip non fourth frame
                    continue
            
            #select targets
            labels = targets[targets[:, 0] == si, 1:]
            
            #native-space targets
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            labelsn = [labelsni for labelsni in labelsn if labelsni[1:].any()]
            labelsn = torch.cat(labelsn, 0).reshape(-1, 5) if len(labelsn) > 0 else torch.zeros((0, 5)).to(device)
            
            nl = len(labelsn) 
            
            tcls = labelsn[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]

            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                else:
                    continue
                
            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                #plot_val_images_for_me(img[si], [path], shapes[si], labelsn[:, 1:], predn[:, :4])
                #print(f"labels {labelsn} predn {predn}")
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
                
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            if save_aot_predictions:
                save_aot_one_pkl(predn, path, False)
            if save_json_gt:
                save_one_json_with_gt(predn, path, class_map, labelsn)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, img[si])

        

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()
        
        # if (batch_i+1)%10 == 0:
        #     print("Ending the batch early, temporary debug please remove this for full training")
        #     break

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4 + '%11i'  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map, epoch_number))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i], epoch_number))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')
    
    #dump results
    if not training:
        #dump ap, recall bla bla so that you don't have to memorize
        with open( str(save_dir / "results.txt"), 'w') as f:
            pf = '%20s' + '%11i' * 2 + '%11.3g' * 4 + '%11i'
            f.writelines([s, "\n", pf % ('all', seen, nt.sum(), mp, mr, map50, map, epoch_number)])
        

    # Save JSON
    
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.pkl")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')

        with open(pred_json, 'wb') as f:
            pickle.dump(jdict, f)
        print("Predictions dumped")
        
        
        
        
        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    

    #Save AOT results
    if save_aot_predictions:
        save_aot_one_pkl(False, False, save_dir / 'aotpredictions' / ('predictions_split_'+ str(get_data_split_number(data_yaml_path)) + '.pkl'))
    if save_json_gt:
        save_one_json_with_gt_dump(save_dir / 'predictionsgt' / ('predictionsgt_split_'+ str(get_data_split_number(data_yaml_path)) + '.pkl'))
    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--save-json-gt', action='store_true', help='save a prediction to JSON results file with gt')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    
    #added by Tushar
    parser.add_argument('--num-frames', type=int, default=5, help='Num of frames to load')
    parser.add_argument('--every-fourth-frame', action='store_true', help='Record results on every fourth frame')
    parser.add_argument('--save-aot-predictions', action='store_true', help='Store predictions in AOT style') 
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    
    if opt.task in ('train', 'val', 'test'):  # run normally
        #print(f"Running task {opt.task}, data yaml {opt.data}, num_frames {opt.num_frames}, save aot_predictions ? 
        # {opt.save_aot_predictions}, test every_fourth_frame {opt.every_fourth_frame}")
        #print(f"Conf-threshold for nms {opt.conf_thres}, iou thresh {opt.iou_thres}")
        run(**vars(opt))

    elif opt.task == 'speed':  # speed benchmarks
        # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=.25, iou_thres=.45,
                device=opt.device, save_json=False, plots=False)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                LOGGER.info(f'\nRunning {f} point {i}...')
                r, _, t = run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                              iou_thres=opt.iou_thres, device=opt.device, save_json=opt.save_json, plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
