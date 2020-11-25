# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
import numpy as np
from tqdm import tqdm

import settings
import model
import dataset

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])


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
        input_shape=(settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, 3),
        input_tensor=input_tensor,
        length=settings.CHAR_SET_LEN * settings.CAPTCHA_LENGTH,
    )

    batch_size = 128
    trainset, train_size = data_loader.get_trainset(batch_size=batch_size)
    testset, test_size = data_loader.get_testset()
    net.train(
        trainset, batch_size=batch_size, epochs=30,
    )

    net.evaluate(testset)

    # correct = 0
    # cnt = 0

    # images, labels = [], []
    # for idx, (image, label) in enumerate(testset):
    #     if idx == test_size:
    #         break

    #     images.append(image)
    #     labels.append(label)


    
    # predictions = net.predict(np.array(images))
    # for prediction, label in tqdm(zip(predictions, labels)):
    #     p = decode_prediction(prediction[0])
    #     l = label[0]
    #     if p == l:
    #         correct += 1

    # test_acc = 100.0 * correct / test_size
    # print(f"[*] Accuracy: {correct}/{test_size} ({test_acc} %)")


if __name__ == "__main__":
    main()
