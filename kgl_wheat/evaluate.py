import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tqdm import tqdm

from kgl_wheat import config
from kgl_wheat.dataset import get_dataset
from kgl_wheat.efficientdet.model import efficientdet
from kgl_wheat.metric import calculate_image_precision
from kgl_wheat.utils import postprocess_bboxes, read_train_csv, get_train_val_split


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
        score_threshold=config.SCORE_THRESHOLD
    )

    prediction_model.load_weights(config.MODEL_WEIGHTS_PATH, by_name=True)

    pred_bboxes, pred_scores, pred_labels = prediction_model.predict(val_dataset, verbose=1)

    pred_bboxes, pred_scores, pred_labels = np.squeeze(pred_bboxes), np.squeeze(pred_scores), np.squeeze(pred_labels)

    pred_bboxes = postprocess_bboxes(bboxes=pred_bboxes, height=config.IMAGE_SIZE, width=config.IMAGE_SIZE)

    pred_bboxes_filtered = []
    pred_scores_filtered = []
    for image_pred_bboxes, image_pred_scores in zip(pred_bboxes, pred_scores):
        indices = image_pred_scores > config.SCORE_THRESHOLD
        pred_bboxes_filtered.append(image_pred_bboxes[indices])
        pred_scores_filtered.append(image_pred_scores[indices])

    pred_bboxes = np.array(pred_bboxes_filtered)
    pred_scores = np.array(pred_scores_filtered)

    precisions = []

    for image_val_bboxes, image_pred_bboxes, image_pred_scores in tqdm(zip(val_bboxes, pred_bboxes, pred_scores), total=len(val_bboxes)):
        sorted_idx = np.argsort(image_pred_scores)[::-1]
        image_pred_bboxes_sorted = image_pred_bboxes[sorted_idx]
        precision = calculate_image_precision(
            gts=image_val_bboxes,
            preds=image_pred_bboxes_sorted,
            thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
            form='coco'
        )
        precisions.append(precision)

    average_precision = np.mean(precisions)

    print('Average validation precision:', average_precision)
