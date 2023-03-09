# EECE 570 Course Project -- Training Neural Networks for Web Page Segmentation

## Acknowledgements

1. The scripts for fine-tuning the HTC model
   :
   The python script is used to make changes to the base configuration for my specific training goal.
   After modifying the training configuration, it fine-tunes a model based on the modified configuration.
   This script is based on the tutorials given by the mmdetection team. The original tutorial notebooks can be found at
   https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_InstanceSeg_Tutorial.ipynb
   and
   https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_Tutorial.ipynb.
2. The scripts used for make inferences from the fine-tuned HTC model:
   These python scripts are based on the relevant scripts given in the following repository:
   https://github.com/webis-de/ecir21-an-empirical-comparison-of-web-page-segmentation-algorithms
   Reference:
   Kiesel, Johannes, et al. "An empirical comparison of web page segmentation algorithms."
   Advances in Information Retrieval:
   43rd European Conference on IR Research, ECIR 2021, Virtual Event, March 28–April 1, 2021, Proceedings, Part II 43.
   Springer International Publishing, 2021.
3. The "cikm20" subdirectory: This folder is a clone of the repository found
   at https://github.com/webis-de/cikm20-web-page-segmentation-revisited-evaluation-framework-and-dataset. It contains
   the code for analysis and post-process the JSON files recording the web page segmentations.
   Reference:
   Kiesel, Johannes, et al. "Web page segmentation revisited: evaluation framework and dataset." Proceedings of the 29th
   ACM International Conference on Information & Knowledge Management. 2020.
4. The htc/mmdet-configs folder: This directory is used to record all the default and modified model configs of the
   mmdetection framework.
   The configuration files in this directory are copied from or based on the files in the following repository:
   https://github.com/open-mmlab/mmdetection/tree/master/configs.
5. The htc/customized-configs folder: This directory is used to record the modified HTC model configuration of the
   mmdetection framework.
   The configuration file in this directory is based on the files in the following repository:
   https://github.com/open-mmlab/mmdetection/tree/master/configs.
6. The htc/checkpoints folder: This directory is used to store the pre-trained model checkpoint.
The pre-trained model checkpoint can be downloaded from https://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_16x1_20e_coco/htc_x101_64x4d_fpn_16x1_20e_coco_20200318-b181fd7a.pth.
I use this check point as the starting point for fine-tuning the model for web page segmentation.
This pretrained checkpoint is the work of the mmdetection team.
7. The yolox directory: This directory contains the files used for training yolox models and making inferences with trained
yolox models. The code used here are based on the code given in the official yolox Github repository:
https://github.com/Megvii-BaseDetection/YOLOX.
More specifically, the demo.py, eval.py, and train.py scripts are taken directly from the official
yolox directory and used. To use these scripts, one needs first to install the yolox packages as specified
in the README.md of the official Github repository.
8. The vips folder: This folder contains the segmentations generated by the VIPS algorithm.
The implementation used is retrieved from this repository:
https://github.com/webis-de/ecir21-an-empirical-comparison-of-web-page-segmentation-algorithms.
Reference:
   * Kiesel, Johannes, et al. "An empirical comparison of web page segmentation algorithms."
   Advances in Information Retrieval:
   43rd European Conference on IR Research, ECIR 2021, Virtual Event, March 28–April 1, 2021, Proceedings, Part II 43.
   Springer International Publishing, 2021.
   * Cai, Deng, et al. "Vips: a vision-based page segmentation algorithm." (2003).