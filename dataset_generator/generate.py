import subprocess as sp
import random
import pathlib
import urllib.request
import concurrent.futures

KCAPTCHA_DIR = "kcaptcha"
PORT = 9999

DATASET_DIR = ".data"
TRAINSET_DIR = "train"
TESTSET_DIR = "test"
VALIDATIONSET_DIR = "validation"
DATASET_SIZE = 50000
TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2
TRAINSET_SIZE = int(DATASET_SIZE * TRAIN_RATIO)
TESTSET_SIZE = int(DATASET_SIZE * TEST_RATIO)
VALIDATIONSET_SIZE = int(DATASET_SIZE * VALIDATION_RATIO)


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


def generate_data(count, download_dir, port):
    nums = "0123456789"
    for i in range(count):
        length = random.choices([2, 3, 4, 5, 6], weights=[1, 10, 100, 1000, 10000], k=1)[0]
        target = "".join(random.choices(nums, k=length))
        save_path = download_dir / ("%s_%.5d.png" % (target, i))
        urllib.request.urlretrieve(
            "http://localhost:%d?string=%s" % (port, target), filename=save_path,
        )

    return str(download_dir)


def main():
    proc = run_kcaptcha_server(KCAPTCHA_DIR, PORT)

    dataset_dir = pathlib.Path(DATASET_DIR)
    trainset_dir = dataset_dir / TRAINSET_DIR
    testset_dir = dataset_dir / TESTSET_DIR
    validationset_dir = dataset_dir / VALIDATIONSET_DIR
    if not dataset_dir.is_dir():
        dataset_dir.mkdir()
        trainset_dir.mkdir()
        testset_dir.mkdir()
        validationset_dir.mkdir()

    # targets = ["%.2d" % num for num in range(0, 100)]
    # pool = concurrent.futures.ThreadPoolExecutor()
    # futures = []
    # for target in targets:
    #     pool.submit(
    #         generate_data, target, TRAINSET_SIZE // len(targets), trainset_dir, PORT
    #     )
    #     pool.submit(
    #         generate_data, target, TESTSET_SIZE // len(targets), testset_dir, PORT
    #     )
    #     pool.submit(
    #         generate_data, target, VALIDATIONSET_SIZE // len(targets), validationset_dir, PORT
    #     )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(generate_data, TRAINSET_SIZE, trainset_dir, PORT),
            executor.submit(generate_data, TESTSET_SIZE, testset_dir, PORT),
            executor.submit(generate_data, VALIDATIONSET_SIZE, validationset_dir, PORT),
        ]

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
