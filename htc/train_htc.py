# Acknowledgement:
# This python script is used to make changes to the base configuration for my specific training goal.
# After modifying the training configuration, it fine-tunes a model based on the modified configuration.
# This script is based on the tutorials given by the mmdetection team. The original tutorial notebooks can be found at
# https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_InstanceSeg_Tutorial.ipynb
# and
# https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_Tutorial.ipynb.
from mmcv import Config, mkdir_or_exist
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
import pickle
from mmdet.apis import train_detector
import os

if __name__ == '__main__':

    cfg = Config.fromfile('mmdet-configs/htc/htc_x101_64x4d_fpn_16x1_20e_coco-copy.py')

    cfg.dataset_type = 'COCODataset'

    cfg.data.test.ann_file = 'coco-formatted-info-test.json'
    cfg.data.test.img_prefix = 'webis-webseg-20-screenshots/'
    cfg.data.test.classes = ('webpage-segmentation',)

    cfg.data.train.ann_file = 'coco-formatted-info-train.json'
    cfg.data.train.img_prefix = 'webis-webseg-20-screenshots/'
    cfg.data.train.classes = ('webpage-segmentation',)

    cfg.data.val.ann_file = 'coco-formatted-info-val.json'
    cfg.data.val.img_prefix = 'webis-webseg-20-screenshots/'
    cfg.data.val.classes = ('webpage-segmentation',)

    # modify num classes of the model in box head and mask head
    for dictionary in cfg.model.roi_head.bbox_head:
        dictionary.num_classes = 1
    for dictionary in cfg.model.roi_head.mask_head:
        dictionary.num_classes = 1

    cfg.load_from = 'checkpoints/htc_x101_64x4d_fpn_16x1_20e_coco_20200318-b181fd7a.pth'

    # Set up working dir to save files and logs.
    cfg.work_dir = 'screenshots-batchsize3-rerun'

    batch_size = 3
    cfg.optimizer.lr = (0.01 / 8) * batch_size
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 10
    cfg.data.samples_per_gpu = batch_size
    cfg.workers_per_gpu = batch_size
    cfg.runner.max_epochs = 20

    cfg.evaluation.interval = 2

    cfg.checkpoint_config.interval = 2

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    cfg.device = 'cuda'
    # Because we do not have semantic segmentation, remove the necessary fields configuring semantic segmentation.
    # See the discussion found at https://github.com/open-mmlab/mmdetection/issues/7683.
    cfg.model.roi_head.semantic_roi_extractor = None
    cfg.model.roi_head.semantic_head = None
    cfg.train_pipeline[-1]['keys'] = cfg.train_pipeline[-1]['keys'][0:4]
    cfg.data.train.pipeline[-1]['keys'] = cfg.data.train.pipeline[-1]['keys'][0:4]
    # We can also use tensorboard to log the training process
    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')]
    print(f'Config:\n{cfg.pretty_text}')
    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_detector(cfg.model)

    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mkdir_or_exist(os.path.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)
    fh = open(os.path.join(cfg.work_dir, "final_model.pickle"), 'wb')
    pickle.dump(model, fh)
    fh.close()
