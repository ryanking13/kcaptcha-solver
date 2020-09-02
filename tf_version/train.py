import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
import settings
import model
import dataset


def main():
    data_loader = dataset.KCaptchaDataLoader()
    input_tensor = layers.Input(shape=(settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT, 1))
    net = model.CAPTCHAMobileNet(
        input_tensor=input_tensor, max_digits=settings.CAPTCHA_LENGTH_MAX,
    )

    batch_size = 64
    trainset, train_size = data_loader.load_trainset(batch_size=batch_size)
    valset, val_size = data_loader.load_validationset(batch_size=batch_size)
    net.train(
        trainset=trainset,
        train_size=train_size,
        valset=valset,
        val_size=val_size,
        batch_size=batch_size,
        epochs=5,
    )


if __name__ == "__main__":
    main()
