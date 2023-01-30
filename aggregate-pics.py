# The webis-webseg-20 dataset available online splits files for each data point in different folders. This is the
# python script used to reorganize the dataset so the files for the same data point are stored in the same folder.
import os
import shutil

wd = os.getcwd()
dataset_dirname = "webis-webseg-20-combined"
copy_target_dir = os.path.join(wd, "webis-webseg-20-edgecoarse")
target_file_name = "screenshot-edges-coarse.png"
if not os.path.isdir(os.path.join(wd, dataset_dirname)):
    print("Cannot find the webis-webseg-20-combined dataset directory is the current directory: " + wd)
    exit(1)

for dpid in os.scandir(os.path.join(wd, dataset_dirname)):
    if os.path.isdir(dpid):
        for content in os.scandir(os.path.join(wd, dataset_dirname, dpid.name)):
            if content.name == target_file_name:
                shutil.copy(content, os.path.join(copy_target_dir, dpid.name + ".png"))
