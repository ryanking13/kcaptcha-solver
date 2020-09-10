import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
import numpy as np
import settings
import model
import dataset


def decode_prediction(predicted):

    c0 = settings.CHAR_SET[np.argmax(predicted[0 : settings.CHAR_SET_LEN])]
    c1 = settings.CHAR_SET[
        np.argmax(predicted[settings.CHAR_SET_LEN : 2 * settings.CHAR_SET_LEN])
    ]

    predicted_num = "%s%s" % (c0, c1)
    return predicted_num


def vec2img(vec):
    image.array_to_img(vec).show()


def main():
    data_loader = dataset.KCaptchaDataLoader()
    input_tensor = layers.Input(shape=(settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, 3))
    net = model.CAPTCHAMobileNet(
        input_tensor=input_tensor,
        length=settings.CHAR_SET_LEN * settings.CAPTCHA_LENGTH,
    )

    batch_size = 64
    trainset, train_size = data_loader.get_trainset(batch_size=batch_size)
    testset, test_size = data_loader.get_testset()
    net.train(
        trainset, batch_size=batch_size, epochs=1,
    )

    net.evaluate(testset)

    correct = 0
    for image, label in testset:
        prediction = net.predict(image)
        label = label[0]
        prediction_num = decode_prediction(prediction[0])
        answer_num = data_loader.one_hot_decode(label)
        print(f"Predicted: {prediction_num} / Answer: {answer_num}")
        # break

    test_acc = 100.0 * correct / test_size
    print(f"[*] Accuracy: {correct}/{test_size} ({test_acc} %)")


if __name__ == "__main__":
    main()
