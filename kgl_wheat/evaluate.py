import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tqdm import tqdm

from kgl_wheat import config
from kgl_wheat.train import read_train_csv, get_max_bboxes, get_train_val_split
from kgl_wheat.dataset import get_dataset
from kgl_wheat.efficientdet.model import efficientdet
from kgl_wheat.metric import calculate_image_precision


def postprocess_boxes(boxes, height, width):
    c_boxes = boxes.copy()
    c_boxes[:, 0] = np.clip(c_boxes[:, 0], 0, width - 1)
    c_boxes[:, 1] = np.clip(c_boxes[:, 1], 0, height - 1)
    c_boxes[:, 2] = np.clip(c_boxes[:, 2], 0, width - 1)
    c_boxes[:, 3] = np.clip(c_boxes[:, 3], 0, height - 1)

    c_boxes[:, 2] = c_boxes[:, 2] - c_boxes[:, 0]
    c_boxes[:, 3] = c_boxes[:, 3] - c_boxes[:, 1]
    return c_boxes


if __name__ == '__main__':
    tf.random.set_seed(22)
    np.random.seed(22)

    image_paths, image_bboxes, image_sources = read_train_csv(
        train_csv_path=config.TRAIN_LABELS_FILE,
        train_images_dir=config.TRAIN_IMAGES_DIR
    )

    max_bboxes = get_max_bboxes(image_bboxes)

    train_image_paths, train_image_bboxes, val_image_paths, val_image_bboxes = \
        get_train_val_split(
            image_paths=image_paths,
            image_bboxes=image_bboxes,
            image_sources=image_sources,
            seed=config.SEED,
            train_size=config.TRAIN_SIZE
        )

    val_dataset = get_dataset(
        image_paths=val_image_paths,
        bboxes=val_image_bboxes,
        max_bboxes=max_bboxes
    )

    model, prediction_model = efficientdet(
        num_classes=1,
        weighted_bifpn=True,
        freeze_bn=True,
        score_threshold=0.7
    )

    model.load_weights(config.MODEL_WEIGHTS_PATH)

    boxes, scores, labels = prediction_model.predict(val_dataset, verbose=2)

    boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)

    boxes = postprocess_boxes(boxes=boxes, height=config.IMAGE_SIZE, width=config.IMAGE_SIZE)

    indices = np.where(scores[:] > config.SCORE_THRESHOLD)[0]
    boxes = boxes[indices]

    precisions = []
    for val_boxes, pred_boxes in tqdm(zip(val_image_bboxes, boxes), total=len(val_image_bboxes)):
        precision = calculate_image_precision(
            gts=val_boxes,
            preds=pred_boxes,
            thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
            form='coco'
        )
        precisions.append(precision)

    print('len(precisions):', len(precisions))
    print('precisions[0]:', precisions[0])
    average_precision = np.mean(precisions)

    print('Average validation precision:', average_precision)
