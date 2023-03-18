# The webis-webseg-20 dataset available online splits files for each data point in different folders. This is the
# python script used to reorganize the dataset so the files for the same data point are stored in the same folder.
import os
import shutil
from pathlib import Path

if __name__ == '__main__':
    wd = os.getcwd()
    enclosing_dir = "webis-webseg-20-combined"
    copy_target_dir = os.path.join(wd, "webis-webseg-20-edgesfine")
    Path(copy_target_dir).mkdir(parents=True, exist_ok=True)
    target_file_name = "screenshot-edges-fine.png"
    if not os.path.isdir(os.path.join(wd, enclosing_dir)):
        print("Cannot find the webis-webseg-20 dataset directory is the current directory: " + wd)
        exit(1)

    for dpid in os.scandir(enclosing_dir):
        if os.path.isdir(dpid):
            for content in os.scandir(os.path.join(enclosing_dir, dpid.name)):
                if content.name == target_file_name:
                    shutil.copy(content, os.path.join(copy_target_dir, dpid.name + ".png"))
