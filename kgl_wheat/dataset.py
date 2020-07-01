from typing import List, Tuple

import numpy as np
import tensorflow as tf

from kgl_wheat import config
from kgl_wheat.efficientdet.anchors import (
    AnchorParameters,
    anchor_targets_bbox,
    anchors_for_shape
)


def decode_img(img: tf.Tensor) -> tf.Tensor:
    """
    Decode image, resize it and preprocess.

    :param img: raw image Tensor
    :return: preprocessed image Tensor
    """
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [config.IMAGE_SIZE, config.IMAGE_SIZE])
    img /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img -= mean
    img /= std
    return img


def convert_bbox(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from format (x1,y1,w,h) to format (x1,y1,x2,y2)

    :param bbox: input bbox
    :return: converted bbox
    """
    return tf.stack([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])


def read_image_and_concat_labels(
    file_path: tf.Tensor,
    bboxes: tf.Tensor,
    bbox_classes: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Read and preprocess image to 3D float Tensor. And convert bbox.

    :param file_path: path to face crop file
    :param bboxes: corresponded bounding box Tensor
    :return: pair of preprocessed image Tensor and corresponded label Tensor
    """
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    label = tf.concat([bboxes, bbox_classes], axis=1)
    return img, label


def preprocess_bboxes(bboxes: List[List[int]], classes: List[List[int]]):
    anchor_parameters = AnchorParameters.default
    anchors = anchors_for_shape((config.IMAGE_SIZE, config.IMAGE_SIZE), anchor_params=anchor_parameters)

    classes_preprocessed, bboxes_preprocessed = anchor_targets_bbox(
            anchors,
            np.array(bboxes),
            np.array(classes),
            num_classes=1
        )
    return bboxes_preprocessed, classes_preprocessed


def get_dataset(image_paths: List[str], bboxes: List[List[int]]) -> tf.data.Dataset:
    """
    Create Tensorflow dataset consisted of face crop/label pairs.

    :param image_paths: image paths
    :param bboxes: bboxes of the detected objects
    :return: Tensorflow dataset
    """
    classes = []
    bboxes_coco = []
    for image_bboxes in bboxes:
        classes.append([])
        bboxes_coco.append([])
        for bbox in image_bboxes:
            classes[-1].append(1)
            bboxes_coco[-1].append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

    print('Preprocess bboxes ...')
    bboxes_preprocessed, classes_preprocessed = preprocess_bboxes(bboxes, classes)

    paths_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    bboxes_dataset = tf.data.Dataset.from_tensor_slices(bboxes_preprocessed)
    classes_dataset = tf.data.Dataset.from_tensor_slices(classes_preprocessed)
    dataset = tf.data.Dataset.zip((paths_dataset, bboxes_dataset, classes_dataset))
    dataset = dataset.map(read_image_and_concat_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
