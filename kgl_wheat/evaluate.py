import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tqdm import tqdm

from kgl_wheat import config
from kgl_wheat.train import read_train_csv, get_max_bboxes, get_train_val_split
from kgl_wheat.dataset import get_dataset
from kgl_wheat.efficientdet.model import efficientdet
from kgl_wheat.metric import calculate_image_precision


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


if __name__ == '__main__':
    tf.random.set_seed(22)
    np.random.seed(22)

    image_paths, bboxes, image_sources = read_train_csv(
        train_csv_path=config.TRAIN_LABELS_FILE,
        train_images_dir=config.TRAIN_IMAGES_DIR
    )

    train_image_paths, train_bboxes, val_image_paths, val_bboxes = \
        get_train_val_split(
            image_paths=image_paths,
            image_bboxes=bboxes,
            image_sources=image_sources,
            seed=config.SEED,
            train_size=config.TRAIN_SIZE
        )

    val_dataset = get_dataset(
        image_paths=val_image_paths,
        bboxes=None
    )

    model, prediction_model = efficientdet(
        num_classes=1,
        weighted_bifpn=True,
        freeze_bn=True,
        score_threshold=0.7
    )

    prediction_model.load_weights(config.MODEL_WEIGHTS_PATH, by_name=True)

    pred_bboxes, pred_scores, pred_labels = prediction_model.predict(val_dataset, verbose=1)

    pred_bboxes, pred_scores, pred_labels = np.squeeze(pred_bboxes), np.squeeze(pred_scores), np.squeeze(pred_labels)

    pred_bboxes = postprocess_bboxes(bboxes=pred_bboxes, height=config.IMAGE_SIZE, width=config.IMAGE_SIZE)

    indices = np.where(pred_scores[:] > config.SCORE_THRESHOLD)[0]
    pred_bboxes = pred_bboxes[indices]

    precisions = []

    for image_val_bboxes, image_pred_bboxes in tqdm(zip(val_bboxes, pred_bboxes), total=len(val_bboxes)):
        precision = calculate_image_precision(
            gts=image_val_bboxes,
            preds=image_pred_bboxes,
            thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
            form='coco'
        )
        precisions.append(precision)

    average_precision = np.mean(precisions)

    print('Average validation precision:', average_precision)
