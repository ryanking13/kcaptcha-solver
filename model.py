import datetime

import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications import DenseNet121

# from tensorflow.keras.applications import mobilenet
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import models
from tensorflow.keras import callbacks
from tensorflow.keras import metrics
import numpy as np


class CAPTCHANet:
    def __init__(
        self,
        input_shape=(224, 224, 3),
        captcha_length=1,
        char_classes=1000,
    ):
        self.prediction_length = captcha_length * char_classes
        # self.net = mobilenet_v2.MobileNetV2(
        #     input_shape=input_shape,
        #     input_tensor=input_tensor,
        #     alpha=1.0,
        #     include_top=False,
        #     weights="imagenet",
        #     # weights=None,
        #     pooling="max",
        # )
        self.input_tensor = layers.Input(shape=input_shape)

        self.net = DenseNet121(
            input_shape=input_shape,
            input_tensor=self.input_tensor,
            include_top=False,
            weights="imagenet",
            pooling="max",
        )

        # for layer in self.net.layers:
        #     layer.trainable = False

        fc1 = layers.Dense(1024, activation="relu")(self.net.output)
        fc2 = layers.Dense(self.prediction_length)(fc1)

        prediction = activations.sigmoid(fc2)

        self.model = models.Model(inputs=self.input_tensor, outputs=prediction)

        opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.model.compile(
            optimizer=opt,
            loss="binary_crossentropy",
            metrics=[self._captcha_accuracy(captcha_length, char_classes)],
        )

        # self.model.summary()

    def train(self, trainset, valset, batch_size, epochs, save_path=None):
        
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )

        callbacks = [tensorboard_callback]
        
        if save_path is not None:
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                save_path,
                monitor="captcha_accuracy",
                save_weights_only=True,
                verbose=1,
            )
            callbacks.append(checkpoint_callback)

        self.model.fit(
            x=trainset,
            epochs=epochs,
            validation_data=valset,
            callbacks=callbacks,
        )

    def _captcha_accuracy(self, captcha_length, classes):
        def captcha_accuracy(y_true, y_pred):
            sum_acc = 0
            for i in range(captcha_length):
                _y_true = tf.slice(y_true, [0, i * classes], [-1, classes])
                _y_pred = tf.slice(y_pred, [0, i * classes], [-1, classes])
                sum_acc += metrics.categorical_accuracy(_y_true, _y_pred)
            return sum_acc / captcha_length
            # return 1

        return captcha_accuracy

    def evaluate(self, x, batch_size):
        self.model.evaluate(x, batch_size=batch_size)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        self.model.save(path)
