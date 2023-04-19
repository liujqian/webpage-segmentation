# EECE 570 Course Project -- Training Neural Networks for Web Page Segmentation

Note: This repository contains other repositories by Git submodule. Please ensure that you initialized all the
submodules when cloning this repository (for example, add the -recurse-submodules flag when cloning).

## Obtaining the dataset and preprocess the dataset

The Webis-Webseg-20 dataset used in this project can be downloaded from this
address: <https://zenodo.org/record/3988124#.ZCyTMdITF0J>.
You need to download webis-webseg-20-screenshots.zip, webis-webseg-20-screenshots-edges.zip,
webis-webseg-20-ground-truth.zip, webis-webseg-20-dom-and-nodes.zip and webis-webseg-20-annotations.zip
and extract those zip files. All the above-mentioned zip files extract to the common folder called "webis-webseg-20"
and it contains many subfolders whose names are the web page IDs. You can rename this folder to "
webis-webseg-20-combined"
to work with other pre-processing scripts.

The aggregate-pics.py script contains the code to separate a specific kind of data from the combined dataset. For
example,
it can extract all the screenshots to a single folder and rename all the "screenshot.png" to "<webpage-id>.png."

The get-coco-formatted-annotations.py can transform all the annotations given in the Webis-Webseg-20 dataset to COCO
formatted segmentation annotations and separate the data into a training set, a validation set and a test set.

Due to the short time frame of project, I did not have time to fully organize the scripts used in the project and many
things are hardcoded. One may want to look into the parameters used in the scripts and update them accordingly based
on the file system structure.

To use the dataset for the training of the YOLOX model, create a folder called webis-webseg-20-screenshots under
the yolox/original-yolox-repo/datasets directory. Create a folder called annotations in this new folder and softlink
the generated COCO formatted annotations to the annotations directory, with names of instances_train2017.json,
instances_val2017.json and instances_test2017.json. Then softlink the extracted screenshots three times under the
created webis-webseg-20-screenshots folder, with names train2017, val2017 and test2017. By doing this I reuse
the COCO dataloader provided in the YOLOX repository. Once the dataset folder is correctly set up,you can train the model 
using the training script provided by the official YOLOX repository. Please refer to the readme file
of that repository to train the YOLOX model. 

To use the dataset for the training of the HTC model, just update the variables encoding the paths of the screenshots
folder and the annotations files in the htc/train_htc.py script.

Finally to use the dataset for training the Faster R-CNN model, create a directory called datasets in the
faster-rcnn/customized-faster-rcnn directory. Then create a subfolder webis-webseg-20 in the newly created folder.
Softlink the folder containing all the extracted screenshots to this webis-webseg-20 folder with the name
webis-webseg-20-screenshots. Also softlink all the COCO formated annotations to this webis-webseg-20 folder with the
names coco-formatted-info-{split}.json where {split} can be either train, val or test.

## Acknowledgments

In this course project, I utilized numerous open-source projects found on GitHub to develop my code as a newcomer to
computer vision programming. Below, I provide a structured overview of the high-level acknowledgments, followed by a
detailed list of acknowledgments for specific parts of the project.

1. The mmdetection project: The official repository of this project is https://github.com/open-mmlab/mmdetection. It is
   an object detection framework, and I used the HTC model from this project. I made adjustments to the configuration
   files and followed the official tutorials to fine-tune an HTC model.
2. The "cikm20-web-page-segmentation-revisited-evaluation-framework-and-dataset" project: This repository contains the
   code from the paper listed below. The repository's address
   is https://github.com/webis-de/cikm20-web-page-segmentation-revisited-evaluation-framework-and-dataset. It provides
   multiple utility scripts written in R to post-process all the inferences I get from the models. I used them for many
   tasks, such as flattening bounding boxes and evaluating the segmentation. Reference: Kiesel, Johannes, et al. "Web
   page segmentation revisited: evaluation framework and dataset." Proceedings of the 29th ACM International Conference
   on Information & Knowledge Management. 2020.
3. The "ecir21-an-empirical-comparison-of-web-page-segmentation-algorithms" project: This project contains code for the
   paper listed below. I adapted their scripts to make inferences from mmdetection models. My post-inference-processing
   pipelines in the batch_process_inference_results.py script are also following the post-process steps stated in this
   repository. That is, I followed the post-process procedures outlined in the README.md of this repository to process
   and evaluate the model inferences. The repository's address
   is https://github.com/webis-de/ecir21-an-empirical-comparison-of-web-page-segmentation-algorithms. Reference: Kiesel,
   Johannes, et al. "An empirical comparison of web page segmentation algorithms." Advances in Information Retrieval:
   43rd European Conference on IR Research, ECIR 2021, Virtual Event, March 28–April 1, 2021, Proceedings, Part II 43.
   Springer International Publishing, 2021.
4. The YOLOX project: I used the project code given in the official YOLOX
   repository (https://github.com/Megvii-BaseDetection/YOLOX) to fine-tune a YOLOX model. I used the scripts from this
   repository for fine-tuning and writing a customized inference script based on their demo script. I also wrote my
   experiment description files to configure the fine-tuning process based on the examples given in their repository.
   Reference: Ge, Zheng, et al. "Yolox: Exceeding yolo series in 2021." arXiv preprint arXiv:2107.08430 (2021).
5. The "simple-faster-rcnn-pytorch" project: My Faster RCNN implementation was adapted from Yun Chen's Faster RCNN
   implementation, which is given in this repository: https://github.com/chenyuntc/simple-faster-rcnn-pytorch. I have
   made several changes to the original implementation, including but not limited to adding a new Coco-based dataloader
   for loading the webis-webseg-20 dataset, changing the inference script for generating inferences in a format that is
   compatible with the R scripts provided by Kiesel et al., adding an RPN only implementation and changing the anchor
   base sizes. Reference: Ren, Shaoqing, et al. "Faster r-cnn: Towards real-time object detection with region proposal
   networks." Advances in neural information processing systems 28 (2015).

### Detailed Acknowledgements

1. The scripts for fine-tuning the HTC model (htc/train_htc.py): The python script is used to make changes to the base
   configuration for my specific training goal. After modifying the training configuration, it fine-tunes a model based
   on the modified configuration. This script is based on the tutorials given by the mmdetection team. The original
   tutorial notebooks can be found
   at https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_InstanceSeg_Tutorial.ipynb
   and https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_Tutorial.ipynb.
2. The scripts used for making inferences from the fine-tuned HTC model: These python scripts are based on the relevant
   scripts given in the following
   repository: https://github.com/webis-de/ecir21-an-empirical-comparison-of-web-page-segmentation-algorithms Reference:
   Kiesel, Johannes, et al. "An empirical comparison of web page segmentation algorithms." Advances in Information
   Retrieval: 43rd European Conference on IR Research, ECIR 2021, Virtual Event, March 28–April 1, 2021, Proceedings,
   Part II 43. Springer International Publishing, 2021.
3. The "cikm20" subdirectory: This folder is a clone of the repository found
   at https://github.com/webis-de/cikm20-web-page-segmentation-revisited-evaluation-framework-and-dataset. It contains
   the code for analysis and post-process the JSON files recording the web page segmentation. Reference: Kiesel,
   Johannes, et al. "Web page segmentation revisited: evaluation framework and dataset." Proceedings of the 29th ACM
   International Conference on Information & Knowledge Management. 2020.
4. The htc/mmdet-configs folder: This directory is used to record all the default and modified model configs of the
   mmdetection framework. The configuration files in this directory are copied from or based on the files in the
   following repository: https://github.com/open-mmlab/mmdetection/tree/master/configs.
5. The htc/customized-configs folder: This directory is used to record the modified HTC model configuration of the
   mmdetection framework. The configuration file in this directory is based on the files in the following
   repository: https://github.com/open-mmlab/mmdetection/tree/master/configs.
6. The htc/checkpoints folder: This directory is used to store the pre-trained model checkpoint. The pre-trained model
   checkpoint can be downloaded
   from https://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_16x1_20e_coco/htc_x101_64x4d_fpn_16x1_20e_coco_20200318-b181fd7a.pth.
   I use this checkpoint as the starting point for fine-tuning the model for web page segmentation. This pretrained
   checkpoint is the work of the mmdetection team.
7. The yolox directory: This directory contains the files used for training YOLOX models and making inferences with
   trained YOLOX models. The code used here is based on the code given in the official YOLOX Github
   repository: https://github.com/Megvii-BaseDetection/YOLOX. More specifically, the demo.py, eval.py, and train.py
   scripts in the tools folder are taken directly from the official YOLOX repository and used. The infer.py script is
   based on the tools/demo.py script found in the official repository. The scripts found under the "exps" folder are
   also based on the experiment description files found in the "exps" folder in the YOLOX official repository. To use
   these scripts, one needs first to install the YOLOX packages as specified in the README.md of the official GitHub
   repository.
8. The vips folder: This folder contains the segmentations generated by the VIPS algorithm. The implementation used is
   retrieved from this
   repository: https://github.com/webis-de/ecir21-an-empirical-comparison-of-web-page-segmentation-algorithms.
   Reference:

* Kiesel, Johannes, et al. "An empirical comparison of web page segmentation algorithms." Advances in Information
  Retrieval: 43rd European Conference on IR Research, ECIR 2021, Virtual Event, March 28–April 1, 2021, Proceedings,
  Part II 43. Springer International Publishing, 2021.
* Cai, Deng, et al. "Vips: a vision-based page segmentation algorithm." (2003). Reference: Ge, Zheng, et al. "Yolox:
  Exceeding yolo series in 2021." arXiv preprint arXiv:2107.08430 (2021).

9. The repository linked by git submodule in the faster-rcnn folder: This repository is adapted from Yun Chen's Faster
   RCNN implementation, which is given in this repository: https://github.com/chenyuntc/simple-faster-rcnn-pytorch. I
   have made several changes to the original implementation, including but not limited to adding a new Coco-based
   dataloader for loading the webis-webseg-20 dataset, changing the inference script for generating inferences in a
   format that is compatible with the R scripts provided by Kiesel et al., adding an RPN only implementation and
   changing the anchor base sizes. reference: Ren, Shaoqing, et al. "Faster r-cnn: Towards real-time object detection
   with region proposal networks." Advances in neural information processing systems 28 (2015).
