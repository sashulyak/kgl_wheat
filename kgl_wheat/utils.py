import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def postprocess_bboxes(bboxes, height, width):
    bboxes_preprocessed = []
    for image_bboxes in bboxes:
        c_boxes = image_bboxes.copy()
        c_boxes[:, 0] = np.clip(c_boxes[:, 0], 0, width - 1)
        c_boxes[:, 1] = np.clip(c_boxes[:, 1], 0, height - 1)
        c_boxes[:, 2] = np.clip(c_boxes[:, 2], 0, width - 1)
        c_boxes[:, 3] = np.clip(c_boxes[:, 3], 0, height - 1)

        c_boxes[:, 2] = c_boxes[:, 2] - c_boxes[:, 0]
        c_boxes[:, 3] = c_boxes[:, 3] - c_boxes[:, 1]
        bboxes_preprocessed.append(c_boxes)
    return np.array(bboxes_preprocessed)


def read_train_csv(train_csv_path: str, train_images_dir: str):
    dataframe = pd.read_csv(train_csv_path)
    image_ids = pd.unique(dataframe['image_id']).tolist()
    image_sources = []
    image_bboxes = []
    image_paths = []
    for image_id in image_ids:
        id_items = dataframe[dataframe['image_id'] == image_id]
        image_sources.append(pd.unique(id_items['source']).tolist()[0])
        image_bbox_strings = id_items['bbox'].tolist()
        image_bbox_numbers = []
        for bbox_string in image_bbox_strings:
            image_bbox_numbers.append(json.loads(bbox_string))
        image_bboxes.append(image_bbox_numbers)
        image_paths.append(os.path.join(train_images_dir, f'{image_id}.jpg'))

    return image_paths, image_bboxes, image_sources


def get_train_val_split(image_paths, image_bboxes, image_sources, seed, train_size):
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, val_idx = next(gss.split(image_paths, image_bboxes, image_sources))
    train_image_paths = np.array(image_paths)[train_idx].tolist()
    train_image_bboxes = np.array(image_bboxes)[train_idx].tolist()
    val_image_paths = np.array(image_paths)[val_idx].tolist()
    val_image_bboxes = np.array(image_bboxes)[val_idx].tolist()
    return train_image_paths, train_image_bboxes, val_image_paths, val_image_bboxes


def get_max_bboxes(image_bboxes):
    max_bboxes = 0
    for bboxes in image_bboxes:
        if len(bboxes) > max_bboxes:
            max_bboxes = len(bboxes)
    return max_bboxes
