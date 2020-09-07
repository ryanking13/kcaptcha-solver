import pathlib

NUMBER = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
CHAR_SET = NUMBER
CHAR_SET_LEN = len(CHAR_SET)

CAPTCHA_LENGTH = 2

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 60

TRAIN_DATASET_PATH = (pathlib.Path() / "../dataset_generator/.data/train").resolve()
TEST_DATASET_PATH = (pathlib.Path() / "../dataset_generator/.data/test").resolve()
VALIDATION_DATASET_PATH = (pathlib.Path() / "../dataset_generator/.data/validation").resolve()

