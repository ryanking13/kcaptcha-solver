import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks
import numpy as np
import settings


class CAPTCHAMobileNet:
    def __init__(self, input_tensor=layers.Input(shape=(224, 224, 3)), max_digits=6):
        self.mobilenet = mobilenet_v2.MobileNetV2(
            input_tensor=input_tensor,
            alpha=1.0,
            include_top=False,
            # weights="imagenet",
            weights=None,
            pooling="max",
        )

        prediction = layers.Dense(settings.CHAR_SET_LEN * settings.CAPTCHA_LENGTH)(
            self.mobilenet.output
        )

        # for layer in self.mobilenet.layers:
        #     layer.trainable = False

        self.model = models.Model(inputs=input_tensor, outputs=prediction)
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # self.model.summary()

    def train(self, trainset, batch_size, epochs):
        train_callbacks = [
            callbacks.ModelCheckpoint("./model_checkpoint", monitor="val_loss"),
            callbacks.ProgbarLogger()
        ]
        self.model.fit(
            x=trainset,
            epochs=epochs,
            callbacks=train_callbacks,
        )

    def predict(self, x):
        return self.model.predict(x)