# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse
import pathlib

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
import numpy as np
from tqdm import tqdm

import model
import dataset

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--length", type=int, default=2, help="Length of CAPTCHA (default: %(default)s)")
    parser.add_argument("-w", "--width", type=int, default=96, help="Width(=height) of input image (default: %(default)s)")
    parser.add_argument("--char-set", default="0123456789", help="Available characters for CAPTCHA (default: %(default)s)")
    parser.add_argument("--dataset-root", default="datasets/.data", help="Dataset root directory (default: %(default)s)")
    parser.add_argument("--train", default="train", help="Train dataset directory name (default: %(default)s)")
    parser.add_argument("--validation", default="validation", help="Validation dataset directory name (default: %(default)s)")
    parser.add_argument("--test", default="test", help="Test dataset directory name (default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=5, help="Traning epochs (default: %(default)s)")

    args = parser.parse_args()
    return args
    

def decode_prediction(predicted, char_set):
    l = len(char_set)
    c0 = char_set[np.argmax(predicted[0 : l])]
    c1 = char_set[
        np.argmax(predicted[l : 2 * l])
    ]

    predicted_num = "%s%s" % (c0, c1)
    return predicted_num


def vec2img(vec):
    image.array_to_img(vec).show()


def main():
    args = parse_args()
    base_dir = pathlib.Path(__file__).resolve().parent  # directory where this file is in
    data_loader = dataset.KCaptchaDataLoader(
        trainset_path = (base_dir / args.dataset_root / args.train).resolve(),
        validationset_path = (base_dir / args.dataset_root / args.validation).resolve(),
        testset_path = (base_dir / args.dataset_root / args.train).resolve(),
        captcha_length = args.length,
        available_chars = args.char_set,
        width = args.width,
        height = args.width,
    )

    input_tensor = layers.Input(shape=(args.width, args.width, 3))
    net = model.CAPTCHANet(
        input_shape=(args.width, args.width, 3),
        input_tensor=input_tensor,
        captcha_length=args.length,
        char_classes=len(args.char_set),
    )

    batch_size = 64
    epochs = args.epochs
    trainset, train_size = data_loader.get_trainset(batch_size=batch_size)
    valset = data_loader.get_validationset()
    testset, test_size = data_loader.get_testset()
    net.train(
        trainset, valset, batch_size=batch_size, epochs=epochs,
    )

    # net.evaluate(testset)

    correct = 0
    cnt = 0

    images, labels = [], []
    for idx, (image, label) in enumerate(testset):
        if idx == test_size:
            break

        images.append(image)
        labels.append(label)


    
    predictions = net.predict(np.squeeze(np.array(images)))
    for prediction, label in tqdm(zip(predictions, labels)):
        p = decode_prediction(prediction, args.char_set)
        l = data_loader.one_hot_decode(label[0])
        # print(p, l)
        if p == l:
            correct += 1

    test_acc = 100.0 * correct / test_size
    print(f"[*] Accuracy: {correct}/{test_size} ({test_acc} %)")


if __name__ == "__main__":
    main()
