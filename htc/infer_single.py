# Acknowledgement:
# This python script is based on the relevant scripts given in the following repository:
# https://github.com/webis-de/ecir21-an-empirical-comparison-of-web-page-segmentation-algorithms
# Reference:
# Kiesel, Johannes, et al. "An empirical comparison of web page segmentation algorithms."
# Advances in Information Retrieval:
# 43rd European Conference on IR Research, ECIR 2021, Virtual Event, March 28–April 1, 2021, Proceedings, Part II 43.
# Springer International Publishing, 2021.
import json
import os
from threading import Lock

import mmcv
import numpy as np
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

lock = Lock()


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


def infer(model, imgfile, id):
    model_used = "_trained" if use_trained_model else "_original"
    outfile = open("segmentations/mmdetection/" + str(id) + model_used + ".json", 'w')
    img = mmcv.imread(imgfile)

    lock.acquire()
    result = inference_detector(model, img)
    lock.release()
    # result = (result[0],None)
    # show_result_pyplot(model, img, result, score_thr=0.10)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None

    bboxes = np.vstack(bbox_result)
    # segm_polygon_list = []
    bbox_polygon_list = []

    # if segm_result is not None:
    #     segms = mmcv.concat_list(segm_result)
    #     inds = np.where(bboxes[:, -1] > 0.0)[0]
    #     for i in inds:
    #         mask = segms[i]
    #         try:
    #             left, right, top, bottom = get_segm_bounds(mask)
    #             if left is not None and right is not None and top is not None and bottom is not None:
    #                 segm_polygon_list.append([[[[left.item(), top.item()], [left.item(), bottom.item()],
    #                                             [right.item(), bottom.item()], [right.item(), top.item()],
    #                                             [left.item(), top.item()]]]])
    #         except ValueError:
    #             print()

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
        height=img.shape[0], width=img.shape[1], id=id,
        segmentations=dict(
            mmdetection_bboxes=bbox_polygon_list,
            # mmdetection_segms=segm_polygon_list,
        ),
    )
    json.dump(out_obj, outfile)


# directory = os.fsencode(sys.argv[1])
config_file_original = '/home/liujqian/Documents/repos/mmdetection/configs/htc/htc_x101_64x4d_fpn_16x1_20e_coco.py'
checkpoint_file_original = 'htc/htc_x101_64x4d_fpn_16x1_20e_coco_latest.pth'

config_file_trained = '/htc/customized-configs/htc_x101_64x4d_fpn_16x1_20e_coco_customized.py'
checkpoint_file_trained = '/home/liujqian/Documents/projects/page-segmentation/htc/screenshot-batchsize3/work_dir_fourth_try_full_screenshot/epoch_6.pth'

use_trained_model = True
config_file = config_file_trained if use_trained_model else config_file_original
checkpoint_file = checkpoint_file_trained if use_trained_model else checkpoint_file_original

model = init_detector(config_file, checkpoint_file, device='cuda:0')

id = "009001"
imgfile = os.path.join(
    "/home/liujqian/Documents/projects/page-segmentation/webis-webseg-20-combined",
    str(id), "screenshot.png")
infer(model, imgfile, id)
