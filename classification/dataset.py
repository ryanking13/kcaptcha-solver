import pathlib
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications import densenet
from tensorflow.keras.applications import efficientnet

import numpy as np
import pandas as pd
from PIL import Image


class KCaptchaDataLoader:

    preprocess_func = {
        "mobilenetv2": mobilenet_v2.preprocess_input,
        "densenet121": densenet.preprocess_input,
        "efficientnetb0": efficientnet.preprocess_input
    }

    def __init__(
        self,
        trainset_path,
        testset_path,
        validationset_path,
        captcha_length,
        available_chars,
        width,
        height,
        base_model,
        verbose=True,
    ):
        self.trainset_path = trainset_path
        self.testset_path = testset_path
        self.validationset_path = validationset_path
        self.captcha_length = captcha_length
        self.available_chars = available_chars
        self.available_chars_cnt = len(self.available_chars)
        self.image_width = width
        self.image_height = height
        self.separator = "_"
        self.ext = "png"
        self.verbose = verbose
        self.preproces_func = KCaptchaDataLoader.preprocess_func[base_model]

        self.datagen = image.ImageDataGenerator(
            # rescale=1.0 / 255,
            # preprocess_input does scaling (https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/applications/imagenet_utils.py)
            preprocessing_function=self.preproces_func,
        )
        self.dataset_loaded = False

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

    def one_hot_encode(self, label):
        """
        e.g.) 17 ==> [0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0,]
        """
        vector = np.zeros(self.available_chars_cnt * self.captcha_length, dtype=float)
        for i, c in enumerate(label):
            idx = i * self.available_chars_cnt + int(c)
            vector[idx] = 1.0
        return vector

    def one_hot_decode(self, vec):
        char_pos = vec.nonzero()[0]
        text = []
        for i, c in enumerate(char_pos):
            digit = c % self.available_chars_cnt
            text.append(str(digit))
        return "".join(text)

    def preprocess(self, img_path):
        img = image.load_img(
            img_path,
            target_size=(self.image_height, self.image_width),
            # color_mode="grayscale",
        )
        img = image.img_to_array(img)
        return img

    def load_dataset(self):
        def _load_dataset_from_dir(path):
            dataset_dir = pathlib.Path(path)
            files = dataset_dir.glob(f"*.{self.ext}")

            x, y = [], []
            for f in files:
                filepath = f.as_posix()
                img = self.preprocess(filepath)
                x.append(img)
                label = f.name.split(self.separator)[0]
                y.append(self.one_hot_encode(label))

            # return np.array(x), np.array(y)
            return np.array(x), np.array(y)

        self._log("Loading train set...")
        self.x_train, self.y_train = _load_dataset_from_dir(self.trainset_path)
        self._log("Loading validation set...")
        self.x_val, self.y_val = _load_dataset_from_dir(self.validationset_path)
        self._log("Loading test set...")
        self.x_test, self.y_test = _load_dataset_from_dir(self.testset_path)
        self._log("Done loading datasets")
        self.dataset_loaded = True

    def _get_dataset(self, x, y, batch_size):
        return (
            self.datagen.flow(
                x=x,
                y=y,
                batch_size=batch_size,
                shuffle=True,
            ),
            len(y),
        )

    def get_trainset(self, batch_size=64):
        if not self.dataset_loaded:  # Lazy data loading
            self.load_dataset()
        return self._get_dataset(self.x_train, self.y_train, batch_size)

    def get_validationset(self):
        if not self.dataset_loaded:  # Lazy data loading
            self.load_dataset()
        return (self.preproces_func(self.x_val), self.y_val), len(self.x_val)

    def get_testset(self, batch_size=1):
        if not self.dataset_loaded:  # Lazy data loading
            self.load_dataset()
        return self._get_dataset(self.x_test, self.y_test, batch_size)

    def _log(self, msg):
        if self.verbose:
            print(f"[*] {self.__class__.__name__}: {msg}")
