import json
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

from kgl_wheat import config
from kgl_wheat.dataset import get_dataset
from kgl_wheat.efficientdet.model import efficientdet
from kgl_wheat.efficientdet.losses import smooth_l1, focal


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
        image_paths.append(os.path.join(config.TRAIN_IMAGES_DIR, f'{image_id}.jpg'))

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


if __name__ == '__main__':
    tf.random.set_seed(22)
    np.random.seed(22)

    image_paths, image_bboxes, image_sources = read_train_csv(
        train_csv_path=config.TRAIN_LABELS_FILE,
        train_images_dir=config.TRAIN_IMAGES_DIR
    )

    train_image_paths, train_image_bboxes, val_image_paths, val_image_bboxes = \
        get_train_val_split(
            image_paths=image_paths,
            image_bboxes=image_bboxes,
            image_sources=image_sources,
            seed=config.SEED,
            train_size=config.TRAIN_SIZE
        )

    train_dataset = get_dataset(
        image_paths=train_image_paths,
        bboxes=train_image_bboxes
    )

    val_dataset = get_dataset(
        image_paths=val_image_paths,
        bboxes=val_image_bboxes
    )

    model, prediction_model = efficientdet(
        num_classes=1,
        weighted_bifpn=True,
        freeze_bn=True,
        score_threshold=0.7
    )

    model.load_weights(config.EFFNET_WEIGHTS_PATH, by_name=True)

    for i in range(1, [227, 329, 329, 374, 464, 566, 656][4]):
        model.layers[i].trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        loss={
            'regression': smooth_l1(),
            'classification': focal()
        }
    )

    model.fit(
        x=train_dataset,
        validation_data=val_dataset,
        epochs=20,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=1,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                config.MODEL_WEIGHTS_PATH,
                save_best_only=True,
                monitor='val_loss'
            )
        ]
    )
