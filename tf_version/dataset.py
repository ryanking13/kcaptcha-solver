# code adapted from: https://towardsdatascience.com/building-a-multi-output-convolutional-neural-network-with-keras-ed24c7bc1178

import pathlib
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import settings


class CAPTCHADataGenerator:
    def __init__(self):
        self.train_df = load_trainset()
        self.test_df = load_testset()
        self.validation_df = load_validationset()

    def preprocess(self, img_path):
        img = Image.open(img_path)
        img = np.array(img) / 255.0

        return img

    def generate_images(self, dataset, training=False, batch_size=16):
        images, labels = [], []
        while True:
            for data in dataset.itertuples():
                f = data["file"]
                img = self.preprocess(f)
                images.append(img)
                labels.append(
                    [
                        to_categorical(data[i])
                        for i in range(settings.CAPTCHA_MAX_LENGTH)
                    ]
                )

                # yielding condition
                if len(images) >= batch_size:
                    yield np.array(images), [
                        np.array(labels),
                    ]
                    images, labels = [], []

            if not training:
                break

    def load_trainset(self):
        return self.generate_images(self.train_df)

    def load_testset(self):
        return self.generate_images(self.test_df)

    def load_validationset(self):
        return self.generate_images(self.validation_df)


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
