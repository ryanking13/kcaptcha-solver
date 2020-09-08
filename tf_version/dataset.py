# code adapted from: https://towardsdatascience.com/building-a-multi-output-convolutional-neural-network-with-keras-ed24c7bc1178

import pathlib
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import mobilenet_v2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import settings


class KCaptchaDataLoader:
    def __init__(
        self,
        trainset_path=settings.TRAIN_DATASET_PATH,
        testset_path=settings.TEST_DATASET_PATH,
        validationset_path=settings.VALIDATION_DATASET_PATH,
    ):
        self.trainset_path = trainset_path
        self.testset_path = testset_path
        self.validationset_path = validationset_path
        self.separator = "_"
        self.ext = ".png"

        self.datagen = image.ImageDataGenerator(
            rescale=1.0 / 255,
            preprocessing_function=mobilenet_v2.preprocess_input,
        )
        self.dataset_loaded = False

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

        # self.train_df = self.dataset_files_to_df(trainset_path)
        # self.test_df = self.dataset_files_to_df(testset_path)
        # self.trainset_generator, self.trainset_size = self.df_to_generator(
        #     self.train_df
        # )
        # self.testset_generator, self.testset_size = self.df_to_generator(self.test_df)

    def one_hot_encode(self, label):
        vector = np.zeros(settings.CHAR_SET_LEN * settings.CAPTCHA_LENGTH, dtype=float)
        for i, c in enumerate(label):
            idx = i * settings.CHAR_SET_LEN + int(c)
            vector[idx] = 1.0
        return vector

    def one_hot_decode(self, vec):
        char_pos = vec.nonzero()[0]
        text = []
        for i, c in enumerate(char_pos):
            digit = c % settings.CHAR_SET_LEN
            text.append(str(digit))
        return "".join(text)

    def preprocess(self, img_path):
        img = image.load_img(
            img_path, target_size=(settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT)
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

            return x, y

        self.x_train, self.y_train = _load_dataset_from_dir(self.trainset_path)
        self.x_val, self.y_val = _load_dataset_from_dir(self.validationset_path)
        self.x_test, self.y_test = _load_dataset_from_dir(self.testset_path)
        self.dataset_loaded = True

    def _get_dataset(x, y, batch_size):
        if not self.dataset_loaded:  # Lazy data loading
            self.load_dataset()

        return (
            self.datagen.flow(
                x=self.x,
                y=self.y,
                batch_size=batch_size,
                shuffle=True,
            ),
            len(self.y),
        )        

    def get_trainset(self, batch_size=64):
        self._get_dataset(x_train, y_train, batch_size)

    def get_validationset(self, batch_size=64):
        self._get_dataset(x_val, y_val, batch_size)

    def get_testset(self, batch_size=64):
        self._get_dataset(x_test, y_test, batch_size)


    # def dataset_files_to_df(self, path, ext=".png", separator="_"):
    #     """generate Pandas DataFrame for dataset generator"""

    #     dataset_dir = pathlib.Path(path)
    #     files = dataset_dir.glob(f"*.{ext}")

    #     labels = []
    #     filepaths = []
    #     for f in files:
    #         filepaths.append(f.as_posix())
    #         label = f.name.split(separator)[0]
    #         labels.append(self.one_hot_encode(label))

    #     df = pd.Dataframe(
    #         {
    #             "file": filepaths,
    #             "label": labels,
    #         }
    #     )
    #     df = df.dropna()

    #     return df

    # def load_dataset(self, dataset, batch_size=64, subset="training"):
    #     x, y = [], []

    #     for data in dataset.itertuples():
    #         f = data.file
    #         x.append(image.img_to_array(image.load_img(f)))
    #         y.append(
    #             [to_categorical(data[i]) for i in range(settings.CAPTCHA_LENGTH_MAX)]
    #         )

    #     datagen = image.ImageDataGenerator(
    #         rescale=1.0 / 255,
    #         validation_split=0.2,
    #         preprocessing_function=mobilenet_v2.preprocess_input,
    #     )

    #     datagen.fit(x)
    #     return (
    #         datagen.flow(
    #             x=x,
    #             y=y,
    #             batch_size=batch_size,
    #             shuffle=True,
    #             subset=subset,
    #         ),
    #         len(x),
    #     )

    # def load_trainset(self, batch_size=64):
    #     return self.load_dataset(
    #         self.train_df, batch_size=batch_size, subset="training"
    #     )

    # def load_testset(self, batch_size=64):
    #     return self.load_dataset(
    #         self.test_df,
    #         batch_size=batch_size,
    #     )

    # def load_validationset(self, batch_size=64):
    #     return self.load_dataset(
    #         self.train_df, batch_size=batch_size, subset="validation"
    #     )


def plot_distribution(pd_series):
    labels = pd_series.value_counts().index.tolist()
    counts = pd_series.value_counts().values.tolist()

    pie_plot = go.Pie(labels=labels, values=counts, hole=0.3)
    fig = go.Figure(data=[pie_plot])
    fig.update_layout(title_text="Distribution for %s" % pd_series.name)

    fig.show()