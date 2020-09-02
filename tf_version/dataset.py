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
    def __init__(self):
        self.train_df = load_trainset()
        self.test_df = load_testset()
        self.validation_df = load_validationset()

    def preprocess(self, img_path):
        img = tf.keras.preprocessing.image.load_img(
            img_path, target_size=(settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT)
        )
        img = tf.keras.preprocessing.image.img_to_array(img)
        return img

    def load_dataset(self, dataset, batch_size=64, subset="training"):
        x, y = [], []

        for data in dataset.itertuples():
            f = data.file
            x.append(image.img_to_array(image.load_img(f)))
            y.append(
                [to_categorical(data[i]) for i in range(settings.CAPTCHA_LENGTH_MAX)]
            )

        datagen = image.ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2,
            preprocessing_function=mobilenet_v2.preprocess_input,
        )

        datagen.fit(x)
        return (
            datagen.flow(x=x, y=y, batch_size=batch_size, shuffle=True, subset=subset,),
            len(x),
        )

    def load_trainset(self, batch_size=64):
        return self.load_dataset(
            self.train_df, batch_size=batch_size, subset="training"
        )

    def load_testset(self, batch_size=64):
        return self.load_dataset(self.test_df, batch_size=batch_size,)

    def load_validationset(self, batch_size=64):
        return self.load_dataset(
            self.train_df, batch_size=batch_size, subset="validation"
        )


def parse_dataset(dataset_path, ext="png"):
    def parse_info_from_file(path, separator="_"):
        fname = path.name
        label_str = fname.split(separator)[0]
        label = [10, 10, 10, 10, 10, 10]
        for i, s in enumerate(label_str):
            label[i] = int(s)
        return tuple(label)

    dataset_dir = pathlib.Path(dataset_path)
    files = dataset_dir.glob(f"*.{ext}")

    records = []
    filenames = []
    for f in files:
        filenames.append(f.as_posix())
        info = parse_info_from_file(f)
        records.append(info)

    df = pd.DataFrame(records)
    df["file"] = filenames
    df = df.dropna()

    return df


def plot_distribution(pd_series):
    labels = pd_series.value_counts().index.tolist()
    counts = pd_series.value_counts().values.tolist()

    pie_plot = go.Pie(labels=labels, values=counts, hole=0.3)
    fig = go.Figure(data=[pie_plot])
    fig.update_layout(title_text="Distribution for %s" % pd_series.name)

    fig.show()


def load_trainset():
    return parse_dataset(settings.TRAIN_DATASET_PATH)


def load_testset():
    return parse_dataset(settings.TEST_DATASET_PATH)


def load_validationset():
    return parse_dataset(settings.VALIDATION_DATASET_PATH)
