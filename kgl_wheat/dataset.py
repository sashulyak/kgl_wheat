from typing import List, Tuple

import numpy as np
import tensorflow as tf

import config


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


def read_image_and_convert_bboxes(
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
    bboxes_converted = tf.map_fn(convert_bbox, bboxes)
    label = tf.concat([bboxes_converted, bbox_classes], axis=1)
    return img, label


def get_dataset(image_paths: List[str], bboxes: List[List[int]], max_bboxes: int) -> tf.data.Dataset:
    """
    Create Tensorflow dataset consisted of face crop/label pairs.

    :param image_paths: image paths
    :param bboxes: bboxes of the detected objects
    :param max_bboxes: max bboxes per image
    :return: Tensorflow dataset
    """
    bboxes_padded = np.zeros(shape=(len(bboxes), max_bboxes, 4), dtype=np.int32)
    classes = np.zeros(shape=(len(bboxes), max_bboxes, 1), dtype=np.int32)
    for i, image_bboxes in enumerate(bboxes):
        for j, bbox in enumerate(image_bboxes):
            bboxes_padded[i, j] = bbox
            classes[i, j, 0] = 1
    paths_datasert = tf.data.Dataset.from_tensor_slices(image_paths)
    bboxes_dataset = tf.data.Dataset.from_tensor_slices(bboxes_padded)
    classes_dataset = tf.data.Dataset.from_tensor_slices(classes)
    dataset = tf.data.Dataset.zip((paths_datasert, bboxes_dataset, classes_dataset))
    dataset = dataset.map(read_image_and_convert_bboxes, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
