import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import argparse
import pathlib

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
import numpy as np
from tqdm import tqdm

import model
import dataset

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        default=2,
        help="Length of CAPTCHA (default: %(default)s)",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=96,
        help="Width(=height) of input image (default: %(default)s)",
    )
    parser.add_argument(
        "--char-set",
        default="0123456789",
        help="Available characters for CAPTCHA (default: %(default)s)",
    )
    parser.add_argument(
        "--train",
        default="datasets/.data/train",
        help="Train dataset directory (default: %(default)s)",
    )
    parser.add_argument(
        "--validation",
        default="datasets/.data/validation",
        help="Validation dataset directory (default: %(default)s)",
    )
    parser.add_argument(
        "--test",
        default="datasets/.data/test",
        help="Test dataset directory (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Traning epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Save best model to specified path, if not specified, model is not saved",
    )

    args = parser.parse_args()
    return args


def decode_prediction(predicted, char_set):
    l = len(char_set)
    c0 = char_set[np.argmax(predicted[0:l])]
    c1 = char_set[np.argmax(predicted[l : 2 * l])]

    predicted_num = "%s%s" % (c0, c1)
    return predicted_num


def vec2img(vec):
    image.array_to_img(vec).show()


def main():
    args = parse_args()

    base_dir = (
        pathlib.Path(__file__).resolve().parent
    )  # directory where this file is in

    trainset_path = (base_dir / args.train).resolve()
    validationset_path = (base_dir / args.validation).resolve()
    testset_path = (base_dir / args.test).resolve()
    batch_size = args.batch_size
    epochs = args.epochs
    captcha_length = args.length
    available_chars = args.char_set
    width = args.width
    height = args.width
    verbose = args.verbose

    if verbose:
        # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        print("-----------------------------------------")
        print(f"Train Set: {trainset_path}")
        print(f"Validation Set: {validationset_path}")
        print(f"Test Set: {testset_path}")
        print(f"Batch Size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Length of CAPTCHA: {captcha_length}")
        print(f"Available Characters: {available_chars}")
        print(f"Image size: {width}x{height}")
        print("-----------------------------------------")

    data_loader = dataset.KCaptchaDataLoader(
        trainset_path=trainset_path,
        validationset_path=validationset_path,
        testset_path=testset_path,
        captcha_length=captcha_length,
        available_chars=available_chars,
        width=width,
        height=height,
    )

    net = model.CAPTCHANet(
        input_shape=(width, height, 3),
        captcha_length=captcha_length,
        char_classes=len(available_chars),
    )

    trainset, train_size = data_loader.get_trainset(batch_size=batch_size)
    valset, val_size = data_loader.get_validationset()
    testset, test_size = data_loader.get_testset()

    if verbose:
        print(f"Train Set Size: {train_size}")
        print(f"Validation Set Size: {val_size}")
        print(f"Test Set Size: {test_size}")

    net.train(
        trainset,
        valset,
        batch_size=batch_size,
        epochs=epochs,
        save_path=args.output,
    )

    # net.model.load_weights("save.hdf5")

    if not verbose:
        net.evaluate(testset, batch_size=batch_size)
    else:
        correct = 0

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
            else:
                print(f"[Wrong] {l} <==> {p} (true/predicted)")

        test_acc = 100.0 * correct / test_size
        print(f"[*] Accuracy: {correct}/{test_size} ({test_acc} %)")


if __name__ == "__main__":
    main()
