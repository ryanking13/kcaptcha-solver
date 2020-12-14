import os

NUMBER = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
CHAR_SET = NUMBER
CHAR_SET_LEN = len(CHAR_SET)
CAPTCHA_LENGTH = 2

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160

TRAIN_DATASET_PATH = os.path.join("dataset_generator", ".data", "train")
TEST_DATASET_PATH = os.path.join("dataset_generator", ".data", "test")
VALIDATION_DATASET_PATH = os.path.join("dataset_generator", ".data", "validation")

