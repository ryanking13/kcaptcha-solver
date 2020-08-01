import subprocess as sp
import pathlib
import urllib.request
import secrets
import concurrent.futures

KCAPTCHA_DIR = "kcaptcha"
PORT = 9999
DATASET_SIZE = 100000
DATASET_DIR = ".data"


def run_kcaptcha_server(docroot, port):
    return sp.Popen(
        ["php", "-S", "localhost:%d" % port, "-t", docroot],
        stdout=sp.DEVNULL,
        stderr=sp.DEVNULL,
    )


def generate_data(target, count, download_dir, port):
    for i in range(count):
        save_path = download_dir / ("%s_%s.png" % (target, secrets.token_hex(16)))
        urllib.request.urlretrieve(
            "http://localhost:%d?string=%s" % (port, target), filename=save_path,
        )

    return target


def main():
    proc = run_kcaptcha_server(KCAPTCHA_DIR, PORT)

    dataset_dir = pathlib.Path(DATASET_DIR)
    if not dataset_dir.is_dir():
        dataset_dir.mkdir()

    targets = ["%.2d" % num for num in range(0, 100)]
    pool = concurrent.futures.ThreadPoolExecutor()
    futures = []
    for target in targets:
        pool.submit(
            generate_data, target, DATASET_SIZE // len(targets), dataset_dir, PORT
        )

    try:
        for completed in concurrent.futures.as_completed(futures):
            print("[+] Done - %d" % completed.result())
    except KeyboardInterrupt:
        # https://gist.github.com/clchiou/f2608cbe54403edb0b13
        pool._threads.clear()
        concurrent.futures.thread._threads_queues.clear()
        raise

    pool.shutdown()


if __name__ == "__main__":
    main()
