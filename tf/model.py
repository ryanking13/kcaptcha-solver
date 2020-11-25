import datetime

import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2

# from tensorflow.keras.applications import mobilenet
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import models
from tensorflow.keras import callbacks
import numpy as np

import settings


class CAPTCHAMobileNet:
    def __init__(
        self,
        input_shape=None,
        input_tensor=layers.Input(shape=(224, 224, 3)),
        length=1000,
    ):
        self.mobilenet = mobilenet_v2.MobileNetV2(
            # self.mobilenet = mobilenet.MobileNet(
            input_shape=input_shape,
            input_tensor=input_tensor,
            alpha=1.0,
            include_top=False,
            weights="imagenet",
            # weights=None,
            pooling="max",
        )

        # for layer in self.mobilenet.layers:
        #     layer.trainable = False

        fc1 = layers.Dense(1024)(self.mobilenet.output)
        fc2 = layers.Dense(length)(fc1)

        prediction = activations.sigmoid(fc2)

        self.model = models.Model(inputs=input_tensor, outputs=prediction)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.model.compile(
            optimizer=opt,
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        # self.model.summary()

    def train(self, trainset, batch_size, epochs):
        # train_callbacks = [
        #     callbacks.ModelCheckpoint("./model_checkpoint", monitor="val_loss"),
        #     callbacks.ProgbarLogger()
        # ]
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )

        self.model.fit(
            x=trainset,
            epochs=epochs,
            # callbacks=train_callbacks,
            # validation_split=0.2,
            callbacks=[tensorboard_callback],
        )

    def evaluate(self, testset):
        self.model.evaluate(testset)

    def predict(self, x):
        return self.model.predict(x)