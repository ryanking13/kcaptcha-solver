from argparse import ArgumentParser
import subprocess as sp
import random
import pathlib
import shutil
import urllib.request
import concurrent.futures
from PIL import Image
import preprocess

KCAPTCHA_DIR = "kcaptcha"
PORT = 9999

DATASET_DIR = ".data"
TRAINSET_DIR = "train"
TESTSET_DIR = "test"
VALIDATIONSET_DIR = "validation"
DATASET_SIZE = 50000
TRAIN_TEST_RATIO = 0.8
NUM_DIGITS = 2


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-s",
        "--dataset-size",
        default=DATASET_SIZE,
        help="Dataset size (Default: %(default)s",
        type=int,
    )
    parser.add_argument(
        "-d",
        "--dataset-dir",
        default=DATASET_DIR,
        help="Directory where dataset will be saved (Default: %(default)s)",
    )
    parser.add_argument(
        "-r",
        "--train-test-ratio",
        default=TRAIN_TEST_RATIO,
        help="Train/Test set ratio (Default: %(default)s)",
        type=float,
    )
    parser.add_argument(
        "-n",
        "--num-digits",
        default=NUM_DIGITS,
        help="Number of CAPTCHA digits (Default: %(default)s)",
        type=int,
    )
    parser.add_argument(
        "-c",
        "--clear",
        default=False,
        action="store_true",
        help="Remove previous datasets before generation (Default: %(default)s",
    )
    parser.add_argument(
        "--no-split-validation",
        default=False,
        action="store_true",
        help="Do not split train/validation set with different directories (Default: %(default)s)",
    )
    parser.add_argument(
        "-p",
        "--preprocess",
        default=False,
        action="store_true",
        help="Do preprocess before saving image (Default: %(default)s)",
    )

    return parser.parse_args()


def run_kcaptcha_server(docroot, port):
    return sp.Popen(
        ["php", "-S", "localhost:%d" % port, "-t", docroot],
        stdout=sp.DEVNULL,
        stderr=sp.DEVNULL,
    )


# def generate_data(target, count, download_dir, port):
#     for i in range(count):
#         save_path = download_dir / ("%s_%.5d.png" % (target, i))
#         urllib.request.urlretrieve(
#             "http://localhost:%d?string=%s" % (port, target), filename=save_path,
#         )

#     return str(download_dir / target)


def generate_data(count, download_dir, port, length, verbose=True, preprocess_func=None):
    nums = "0123456789"
    dirname = download_dir.name
    for i in range(count):
        # length = random.choices(
        #     [2, 3, 4, 5, 6], weights=[1, 10, 100, 1000, 10000], k=1
        # )[0]
        target = "".join(random.choices(nums, k=length))
        # target = "10"  # for test
        save_path = download_dir / ("%s_%.6d.png" % (target, i))

        if preprocess_func is None:
            urllib.request.urlretrieve(
                "http://localhost:%d?string=%s" % (port, target),
                filename=save_path,
            )
        else:
            content = urllib.request.urlopen("http://localhost:%d?string=%s" % (port, target))
            img = preprocess_func(content)
            img.save(save_path)

        if verbose and i % 1000 == 0:
            print(f"[{dirname}]: {i}")

    return dirname


def preprocess_img(sz=(96, 96)):
    width = 160
    height = 60
    def _preprocess(img):
        img = Image.open(img)
        # img = preprocess.to_grayscale(img)
        img = preprocess.crop_img(img, sz=(min(width, sz[0]), min(height, sz[1])))
        img = preprocess.resize_img(img, sz)
        img = preprocess.filter_img(img)
        return img
    
    return _preprocess


def main():
    args = parse_args()
    proc = run_kcaptcha_server(KCAPTCHA_DIR, PORT)

    dataset_dir = pathlib.Path(args.dataset_dir)
    trainset_dir = dataset_dir / TRAINSET_DIR
    testset_dir = dataset_dir / TESTSET_DIR

    if args.clear and dataset_dir.is_dir():
        shutil.rmtree(str(dataset_dir))

    dataset_dir.is_dir() or dataset_dir.mkdir()
    trainset_dir.is_dir() or trainset_dir.mkdir()
    testset_dir.is_dir() or testset_dir.mkdir()

    trainset_size = int(args.dataset_size * args.train_test_ratio)
    testset_size = args.dataset_size - trainset_size

    if not args.no_split_validation:
        validationset_dir = dataset_dir / VALIDATIONSET_DIR
        validationset_dir.is_dir() or validationset_dir.mkdir()
        # split validation set from trainset
        validationset_size = int(trainset_size * (1 - args.train_test_ratio))
        trainset_size -= validationset_size

    preprocess_func = preprocess_img() if args.preprocess else None
    verbose = True
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                generate_data, trainset_size, trainset_dir, PORT, args.num_digits, verbose, preprocess_func,
            ),
            executor.submit(
                generate_data, testset_size, testset_dir, PORT, args.num_digits, verbose, preprocess_func,
            ),
        ]

        if not args.no_split_validation:
            futures.append(
                executor.submit(
                    generate_data,
                    validationset_size,
                    validationset_dir,
                    PORT,
                    args.num_digits,
                    verbose,
                    preprocess_func,
                ),
            )

        try:
            for completed in concurrent.futures.as_completed(futures):
                print("[+] Done: %s" % completed.result())
        except KeyboardInterrupt:
            # https://gist.github.com/clchiou/f2608cbe54403edb0b13
            executor._threads.clear()
            concurrent.futures.thread._threads_queues.clear()
            raise

    print("[+] Done")


if __name__ == "__main__":
    main()
