import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from yolox.data import ValTransform
from yolox.exp import get_exp
from yolox.utils import get_model_info, vis, postprocess


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=("webpage-segmentation",),
            trt_file=None,
            decoder=None,
            device="cpu",
            fp16=False,
            legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            print("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


if __name__ == '__main__':
    ckpt_file = "checkpoints/edges_coarse_best_ckpt.pth"
    exp_file = "exps/webis_webseg_yolox_l.py"
    train_target_type = "screenshots-edges-coarse"
    inference_target_dir = f"webis-webseg-20-{train_target_type}"

    exp = get_exp(exp_file=exp_file)
    experiment_name = exp.exp_name
    exp.nmsthre = 0.45
    exp.test_size = (640, 640)
    exp.test_conf = 0.10
    model = exp.get_model()
    print("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    predictor = Predictor(
        model, exp, ("webpage-segmentation",), trt_file=None, decoder=None,
        device="cpu", fp16=False, legacy=False,
    )
    results_target_dir = os.path.join("inference_out", train_target_type, "original_inferences")
    Path(results_target_dir).mkdir(parents=True, exist_ok=True)
    ids = [d.name.split(".")[0] for d in os.scandir(inference_target_dir) if int(d.name.split(".")[0]) > 9487]
    for i in range(len(ids)):
        bbox_polygon_list = []
        if i % 50 == 0:
            print(f"Making inference for the {i}th data point. There are {len(ids)} data points in total!")
        img_id = ids[i]
        img_file_name = os.path.join(inference_target_dir, img_id + ".png")
        outputs, img_info = predictor.inference(img_file_name)
        ratio = img_info["ratio"]
        bboxes = outputs[0][:, 0:4]
        bboxes /= ratio
        for bbox in bboxes:
            bbox = bbox.cpu().detach().numpy()
            bbox_int = bbox.astype(np.int32)
            left = bbox_int[0]
            top = bbox_int[1]
            right = bbox_int[2]
            bottom = bbox_int[3]
            bbox_polygon_list.append(
                [
                    [
                        [
                            [
                                left.item(),
                                top.item(),
                            ],
                            [
                                left.item(),
                                bottom.item(),
                            ],
                            [
                                right.item(),
                                bottom.item(),
                            ],
                            [
                                right.item(),
                                top.item(),
                            ],
                            [
                                left.item(),
                                top.item()
                            ]
                        ]
                    ]
                ]
            )
        out_obj = dict(
            height=img_info['height'],
            width=img_info['width'],
            id=img_id,
            segmentations=dict(
                yolox_bboxes=bbox_polygon_list,
            ),
        )
        with open(os.path.join(results_target_dir, img_id + ".json"), 'w') as outfile:
            json.dump(out_obj, outfile)
