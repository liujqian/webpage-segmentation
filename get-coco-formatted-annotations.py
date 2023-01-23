# This script is used to transform and combine all the annotations of the webis-webseg-20 dataset to the coco annotation
# format, so it can be used for the mmdetection library.
import os.path

import mmcv

dataset_dirname = "webis-webseg-20"
ground_truth_dir_name = "webis-webseg-20-ground-truth"
annotation_file_name = "ground-truth.json"
polygons_with_holes = []
"""
The coco annotation format:
{
    "images": [image],
    "annotations": [annotation],
    "categories": [category]
}


image = {
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
}

annotation = {
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}

categories = [{
    "id": int,
    "name": str,
    "supercategory": str,
}]
"""


def get_annotations_of_image(img_id: str, next_annotation_id: int) -> (dict, list[dict]):
    filepath = os.path.join(dataset_dirname, ground_truth_dir_name, img_id, annotation_file_name)
    data = mmcv.load(filepath, file_format="json")
    image = {"id": int(img_id), "width": data["width"], "height": data["height"], "file_name": img_id + '.png'}
    segs = data["segmentations"]
    annotations = []
    if "majority-vote" not in segs:
        raise ValueError("The segmentations field of image " + img_id + " doesn't contain a majority-vote sub-field!")
    mv_segs = segs["majority-vote"]
    for mp_id, multipoligon in enumerate(mv_segs):
        for p_id, polygon in enumerate(multipoligon):
            if len(polygon) > 1:
                polygon_info = {"img-id": img_id, "multipoligon-idx": mp_id, "polygon-idx": p_id}
                polygons_with_holes.append(polygon_info)
            ring = polygon[0]
            rezipped_coordinates = [*zip(*ring)]
            xmin, xmax, ymin, ymax = (
                min(rezipped_coordinates[0]),
                max(rezipped_coordinates[0]),
                min(rezipped_coordinates[1]),
                max(rezipped_coordinates[1]),
            )
            annotation = {
                "id": next_annotation_id,
                "image_id": int(img_id),
                "category_id": 0,
                "segmentation": [[x for point in ring for x in point]],
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "area": (xmax - xmin) * (ymax - ymin),
                "iscrowd": 0
            }
            annotations.append(annotation)
            next_annotation_id += 1
    return image, annotations


categories = [
    {
        "id": 0,
        "name": "webpage-segmentation",
    }
]
coco_formatted_info_train = {"categories": categories, "images": [], "annotations": []}
coco_formatted_info_val = {"categories": categories, "images": [], "annotations": []}
next_annotation_id = 0
for folder in os.scandir(os.path.join(dataset_dirname, ground_truth_dir_name)):
    folder_name = folder.name
    img_info, annotations = get_annotations_of_image(folder_name, next_annotation_id)
    next_annotation_id += len(annotations)
    if int(folder_name) > 8999:
        coco_formatted_info = coco_formatted_info_val
    else:
        coco_formatted_info = coco_formatted_info_train
    coco_formatted_info["images"].append(img_info)
    coco_formatted_info["annotations"].extend(annotations)
mmcv.dump(coco_formatted_info_train, "coco-formatted-info-train.json", "json")
mmcv.dump(coco_formatted_info_val, "coco-formatted-info-val.json", "json")
