# EECE 570 Course Project -- Training Neural Networks for Web Page Segmentation

## Acknowledgements

As a newbie in computer vision programming, the code for the experiments of the project is mostly based on
many open-source projects found on Github. In the "Detailed Acknowledgements" section below I tried my best
to give acknowledgements to all the open-source projects which I used or adapted my code from. Here I give
some high-level acknowledgements:

1. The mmdetection project: The official repository of this project is <https://github.com/open-mmlab/mmdetection>.
It is a object detection framework and I used the HTC model from this project. I made adjustments to the configuration flies
and followed the official tutorials to fine-tune a HTC model.
2. The "cikm20-web-page-segmentation-revisited-evaluation-framework-and-dataset" project:
   This repository contains the code from the paper listed below. The repository's address is <https://github.com/webis-de/cikm20-web-page-segmentation-revisited-evaluation-framework-and-dataset>.
    It provides multiple utility scripts written in R to post-process all the inferences I get from the models. I used them for many tasks such as
    to flatten bounding boxes, and to evaluate the segmentations.
   Reference:
   Kiesel, Johannes, et al. "Web page segmentation revisited: evaluation framework and dataset." Proceedings of the 29th
   ACM International Conference on Information & Knowledge Management. 2020.
3. The "ecir21-an-empirical-comparison-of-web-page-segmentation-algorithms" project:
   This project contains code for the paper listed below. I adapted their scripts to make inferences from mmdetection models.
   My post-inference-processing pipelines in the batch_process_inference_results.py script are also following the post-process steps stated in this repository.
   That is, I followed the post-process procedures outlined in the README.md of this repository to process and evaluate the model inferences.
   The repository's address is <https://github.com/webis-de/ecir21-an-empirical-comparison-of-web-page-segmentation-algorithms>.
   Reference:
   Kiesel, Johannes, et al. "An empirical comparison of web page segmentation algorithms."
   Advances in Information Retrieval:
   43rd European Conference on IR Research, ECIR 2021, Virtual Event, March 28–April 1, 2021, Proceedings, Part II 43.
   Springer International Publishing, 2021.
4. The yolox project:
I used the project code given in the official yolox repository (<https://github.com/Megvii-BaseDetection/YOLOX>) to fine-tune a yolox model. I used the scripts from this repository
for fine-tuning and write a customized inference script based on their demo script. I also wrote my experiment description files to configure the fine-tuning process based on the examples
given in their repository.
Reference: Ge, Zheng, et al. "Yolox: Exceeding yolo series in 2021." arXiv preprint arXiv:2107.08430 (2021).
5. The "simple-faster-rcnn-pytorch" project:
My Faster RCNN implementation was adapted from Yun Chen's Faster RCNN implementation, which is given in this repository:
<https://github.com/chenyuntc/simple-faster-rcnn-pytorch>.
I have made several changes to the original implementation, including but not limited adding a new Coco-based dataloader for loading the webis-webseg-20 dataset,
changing the inference script for generating inferences in a format that is compatible with the R scripts provided by Kiesel et al., adding an RPN only implementation and changing the anchor base sizes.
Reference: Ren, Shaoqing, et al. "Faster r-cnn: Towards real-time object detection with region proposal networks." Advances in neural information processing systems 28 (2015).

### Detailed Acknowledgements

1. The scripts for fine-tuning the HTC model (htc/train_htc.py):
   The python script is used to make changes to the base configuration for my specific training goal.
   After modifying the training configuration, it fine-tunes a model based on the modified configuration.
   This script is based on the tutorials given by the mmdetection team. The original tutorial notebooks can be found at
   <https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_InstanceSeg_Tutorial.ipynb>
   and
   <https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_Tutorial.ipynb>.
2. The scripts used for make inferences from the fine-tuned HTC model:
   These python scripts are based on the relevant scripts given in the following repository:
   <https://github.com/webis-de/ecir21-an-empirical-comparison-of-web-page-segmentation-algorithms>
   Reference:
   Kiesel, Johannes, et al. "An empirical comparison of web page segmentation algorithms."
   Advances in Information Retrieval:
   43rd European Conference on IR Research, ECIR 2021, Virtual Event, March 28–April 1, 2021, Proceedings, Part II 43.
   Springer International Publishing, 2021.
3. The "cikm20" subdirectory: This folder is a clone of the repository found
   at <https://github.com/webis-de/cikm20-web-page-segmentation-revisited-evaluation-framework-and-dataset>. It contains
   the code for analysis and post-process the JSON files recording the web page segmentations.
   Reference:
   Kiesel, Johannes, et al. "Web page segmentation revisited: evaluation framework and dataset." Proceedings of the 29th
   ACM International Conference on Information & Knowledge Management. 2020.
4. The htc/mmdet-configs folder: This directory is used to record all the default and modified model configs of the
   mmdetection framework.
   The configuration files in this directory are copied from or based on the files in the following repository:
   <https://github.com/open-mmlab/mmdetection/tree/master/configs>.
5. The htc/customized-configs folder: This directory is used to record the modified HTC model configuration of the
   mmdetection framework.
   The configuration file in this directory is based on the files in the following repository:
   <https://github.com/open-mmlab/mmdetection/tree/master/configs>.
6. The htc/checkpoints folder: This directory is used to store the pre-trained model checkpoint.
The pre-trained model checkpoint can be downloaded from <https://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_16x1_20e_coco/htc_x101_64x4d_fpn_16x1_20e_coco_20200318-b181fd7a.pth>.
I use this check point as the starting point for fine-tuning the model for web page segmentation.
This pretrained checkpoint is the work of the mmdetection team.
7. The yolox directory: This directory contains the files used for training yolox models and making inferences with trained
yolox models. The code used here are based on the code given in the official yolox Github repository:
<https://github.com/Megvii-BaseDetection/YOLOX>.
More specifically, the demo.py, eval.py, and train.py scripts in the tools folder are taken directly from the official
yolox directory and used. The infer.py script is based on the tools/demo.py script found in the official repository. The scripts found under the
exps folder are also based on the experiment description files found in the exps folder in the yolox official repository.
To use these scripts, one needs first to install the yolox packages as specified
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
Reference: Ge, Zheng, et al. "Yolox: Exceeding yolo series in 2021." arXiv preprint arXiv:2107.08430 (2021).

9. The repository linked by git submodule in the faster-rcnn folder:
This repository is adapted from Yun Chen's Faster RCNN implementation, which is given in this repository:
<https://github.com/chenyuntc/simple-faster-rcnn-pytorch>.
I have made several changes to the original implementation, including but not limited adding a new Coco-based dataloader for loading the webis-webseg-20 dataset,
changing the inference script for generating inferences in a format that is compatible with the R scripts provided by Kiesel et al., adding an RPN only implementation and changing the anchor base sizes.
reference: Ren, Shaoqing, et al. "Faster r-cnn: Towards real-time object detection with region proposal networks." Advances in neural information processing systems 28 (2015).
