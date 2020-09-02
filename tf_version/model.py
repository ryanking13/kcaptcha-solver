import tensorflow as tf
import tf.keras.applications.mobilenet_vs as mobilenet_v2
import tf.keras.layers as layers
import tf.keras.models as models
import tf.keras.callbacks as callbacks
import numpy as np
import settings
from dataset import CAPTCHADataGenerator


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

        predictions = [
            layers.Dense(11, activation="softmax")(self.mobilenet.output)
            for _ in range(settings.CAPTCHA_MAX_LENGTH)
        ]

        # for layer in self.mobilenet.layers:
        #     layer.trainable = False

        self.model = models.Model(inputs=input_tensor, outputs=predictions)
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # self.model.summary()

    def train(self, dataset_generator, dataset_length, batch_size, epochs, validation_generator, validation_length):
        callbacks = [
            callbacks.ModelCheckpoint("./model_checkpoint", monitor='val_loss')
        ]
        self.model.fit_generator(
            generator=dataset_generator,
            steps_per_epoch=dataset_length / batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_length / batch_size,
            callbacks=callbacks,
        )

    def predict(self, x):
        return self.model.predict(x)


def main():
    model = CAPTCHAMobileNet()

    img_path = "011078_00204.png"
    img = tf.keras.preprocessing.image.load_img(
        img_path, target_size=(settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT)
    )
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    preds = model.predict(x)
    print(preds)


if __name__ == "__main__":
    main()