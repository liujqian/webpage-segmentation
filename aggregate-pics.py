# The webis-webseg-20 dataset available online splits files for each data point in different folders. This is the
# python script used to reorganize the dataset so the files for the same data point are stored in the same folder.
import os
import shutil

wd = os.getcwd()
dataset_dirname = "webis-webseg-20"
copy_target_dir = os.path.join(wd, "webis-webseg-20-screenshots")
target_file_name = "screenshot.png"
if not os.path.isdir(os.path.join(wd, dataset_dirname)):
    print("Cannot find the webis-webseg-20 dataset directory is the current directory: " + wd)
    exit(1)

for dpid in os.scandir("webis-webseg-20"):
    if os.path.isdir(dpid):
        for content in os.scandir(
                os.path.join("webis-webseg-20", dpid.name)):
            if content.name == target_file_name:
                shutil.copy(content, os.path.join(copy_target_dir, dpid.name + ".png"))
