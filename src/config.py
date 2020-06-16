import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TEST_IMAGES_DIR = os.path.join(BASE_DIR, 'data/test')
TRAIN_IMAGES_DIR = os.path.join(BASE_DIR, 'data/train')
TRAIN_LABELS_FILE = os.path.join(BASE_DIR, 'data/train.csv')
SAMPLE_SUBMISSION_FILE = os.path.join(BASE_DIR, 'data/sample_submission.csv')

IMAGE_SIZE = 1024
TRAIN_SIZE = 0.9
SEED = 42
