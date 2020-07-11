import json
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from kgl_wheat import config
from kgl_wheat.dataset import get_dataset
from kgl_wheat.efficientdet.model import efficientdet
from kgl_wheat.efficientdet.losses import smooth_l1, focal
from kgl_wheat.utils import read_train_csv, get_train_val_split


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
