import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TEST_IMAGES_DIR = os.path.join(BASE_DIR, 'data/test')
TRAIN_IMAGES_DIR = os.path.join(BASE_DIR, 'data/train')
TRAIN_LABELS_FILE = os.path.join(BASE_DIR, 'data/train.csv')
SAMPLE_SUBMISSION_FILE = os.path.join(BASE_DIR, 'data/sample_submission.csv')

EFFNET_WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    'weights/efficientnet-b4_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
)

MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, 'weights/kgl_wheat_model.h5')
MODEL_WEIGHTS_PRED_PATH = os.path.join(BASE_DIR, 'weights/kgl_wheat_model_pred.h5')

IMAGE_SIZE = 1024
BATCH_SIZE = 4
TRAIN_SIZE = 0.9
SEED = 42
SCORE_THRESHOLD = 0.7
