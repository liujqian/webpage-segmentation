# The webis-webseg-20 dataset available online splits files for each data point in different folders. This is the
# python script used to reorganize the dataset so the files for the same data point are stored in the same folder.
import os
import shutil

wd = os.getcwd()
dataset_dirname = "webis-webseg-20"
if not os.path.isdir(os.path.join(wd, dataset_dirname)):
    print("Cannot find the webis-webseg-20 dataset directory is the current directory: " + wd)
    exit(1)
postfixes = ["annotations", "dom-and-nodes", "ground-truth", "screenshots", "screenshots-edges"]
for postfix in postfixes:
    if not os.path.isdir(os.path.join(wd, dataset_dirname, dataset_dirname + "-" + postfix)):
        print("missing the " + postfix + " directory! Please double check!")
        exit(1)
os.makedirs(dataset_dirname + "-combined", exist_ok=True)
for dpid in os.scandir(os.path.join(wd, dataset_dirname, dataset_dirname + "-screenshots", dataset_dirname)):
    if os.path.isdir(dpid):
        dpdir_name = os.path.join(dataset_dirname + "-combined", dpid.name)
        os.makedirs(dpdir_name)
        for postfix in postfixes:
            for content in os.scandir(
                    os.path.join(wd, dataset_dirname, dataset_dirname + "-" + postfix, dataset_dirname, dpid.name)):
                shutil.copy(content, os.path.join(dpdir_name, content.name))
