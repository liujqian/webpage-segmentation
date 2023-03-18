# Acknowledgement:
# This python script is based on the relevant scripts given in the following repository:
# https://github.com/webis-de/ecir21-an-empirical-comparison-of-web-page-segmentation-algorithms
# Reference:
# Kiesel, Johannes, et al. "An empirical comparison of web page segmentation algorithms."
# Advances in Information Retrieval:
# 43rd European Conference on IR Research, ECIR 2021, Virtual Event, March 28–April 1, 2021, Proceedings, Part II 43.
# Springer International Publishing, 2021.
from pathlib import Path

from mmdet.apis import init_detector, inference_detector
import pycocotools.mask as maskUtils
import numpy as np
import mmcv
import sys
import os
import json
from threading import Thread, Lock


def get_segm_left(mask):
    return mask.any(0).argmax()


def get_segm_right(mask):
    return mask.shape[1] - np.fliplr(mask).any(0).argmax()


def get_segm_top(mask):
    return mask.any(1).argmax()


def get_segm_bottom(mask):
    return mask.shape[0] - np.flipud(mask).any(1).argmax()


def get_segm_bounds(mask):
    left = get_segm_left(mask)
    right = get_segm_right(mask)
    top = get_segm_top(mask)
    bottom = get_segm_bottom(mask)
    if left is not None and right is not None and top is not None and bottom is not None:
        return left, right, top, bottom
    else:
        raise ValueError('Could not determine bounds for segment')


lock = Lock()


def infer(model, imgfile, id, train_target_type):
    img_id = id
    target_dir = os.path.join("inference_out", train_target_type, "original_inferences")
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    target_file = os.path.join(target_dir, img_id + ".json")
    img = mmcv.imread(imgfile)

    lock.acquire()
    result = inference_detector(model, img)
    lock.release()

    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None

    bboxes = np.vstack(bbox_result)
    segm_polygon_list = []
    bbox_polygon_list = []

    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > 0.07)[0]
        for i in inds:
            mask = segms[i]
            try:
                left, right, top, bottom = get_segm_bounds(mask)

                if left is not None and right is not None and top is not None and bottom is not None:
                    segm_polygon_list.append([[[[left.item(), top.item()], [left.item(), bottom.item()],
                                                [right.item(), bottom.item()], [right.item(), top.item()],
                                                [left.item(), top.item()]]]])
            except ValueError:
                print()

    for bbox in bboxes:
        if bbox[4] < 0.07:
            continue
        bbox_int = bbox.astype(np.int32)
        left = bbox_int[0]
        top = bbox_int[1]
        right = bbox_int[2]
        bottom = bbox_int[3]

        bbox_polygon_list.append([[[[left.item(), top.item()], [left.item(), bottom.item()],
                                    [right.item(), bottom.item()], [right.item(), top.item()],
                                    [left.item(), top.item()]]]])

    out_obj = dict(
        height=img.shape[0],
        width=img.shape[1],
        id=img_id,
        segmentations=dict(
            mmdetection_bboxes=bbox_polygon_list,
            mmdetection_segms=segm_polygon_list
        ),
    )
    with open(target_file, 'w') as outfile:
        json.dump(out_obj, outfile)


checkpoint_file_trained = 'checkpoints/edge-fine-epoch6/epoch_6.pth'
if __name__ == '__main__':
    train_target_type = "edges-fine"
    inference_target_dir = "../yolox/webis-webseg-20-screenshots-edges-fine"

    directory = inference_target_dir
    config_file_trained = 'customized-configs/htc_x101_64x4d_fpn_16x1_20e_coco_customized.py'
    model = init_detector(config_file_trained, checkpoint_file_trained, device='cpu')

    ids = [d.name for d in os.scandir(directory) if int(d.name.split(".")[0]) > 9487]

    for i in range(len(ids)):
        if i % 50 == 0:
            print(f"Making inference for the {i}th data point. There are {len(ids)} data points in total!")
        img_id = ids[i].split(".")[0]

        infer(
            model,
            os.path.join(inference_target_dir, f"{img_id}.png"),
            img_id,
            train_target_type=train_target_type
        )
