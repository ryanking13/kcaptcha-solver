import os
import hashlib
import pathlib
import json

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import tqdm

flags.DEFINE_string(
    "data_dir",
    ".data/",
    "path to raw KCAPTCHA dataset",
)
flags.DEFINE_enum("split", "train", ["train", "validation", "test"], "specify train or validation or test split")
flags.DEFINE_string("output_file", ".data/kcaptcha_train.tfrecord", "output dataset")

def build_record(image_file, annotation_file):
    img_raw = image_file.read_bytes()
    annotation = json.loads(annotation_file.read_text())
    
    key = hashlib.sha256(img_raw).hexdigest()
    height = annotation["height"]
    width = annotation["width"]
    filename = annotation["path"].split("/")[-1]

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    views = []
    difficult = []

    for bbox in annotation["bbox"]:
        xmin.append(float(bbox["xmin"]) / width)
        xmax.append(float(bbox["xmax"]) / width)
        ymin.append(float(bbox["ymin"]) / height)
        ymax.append(float(bbox["ymax"]) / height)
        classes.append(int(bbox["label"]))
        classes_text.append(bbox["label"].encode("utf8"))
        truncated.append(0)
        views.append(b"")
        difficult.append(0)

    record = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[height])
                ),
                "image/width": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[width])
                ),
                "image/filename": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[filename.encode("utf8")]
                    )
                ),
                "image/source_id": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[filename.encode("utf8")]
                    )
                ),
                "image/key/sha256": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[key.encode("utf8")])
                ),
                "image/encoded": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[img_raw])
                ),
                "image/format": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=["png".encode("utf8")])
                ),
                "image/object/bbox/xmin": tf.train.Feature(
                    float_list=tf.train.FloatList(value=xmin)
                ),
                "image/object/bbox/xmax": tf.train.Feature(
                    float_list=tf.train.FloatList(value=xmax)
                ),
                "image/object/bbox/ymin": tf.train.Feature(
                    float_list=tf.train.FloatList(value=ymin)
                ),
                "image/object/bbox/ymax": tf.train.Feature(
                    float_list=tf.train.FloatList(value=ymax)
                ),
                "image/object/class/text": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=classes_text)
                ),
                "image/object/class/label": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=classes)
                ),
                "image/object/difficult": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=difficult)
                ),
                "image/object/truncated": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=truncated)
                ),
                "image/object/view": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=views)
                ),
            }
        )
    )

    return record


def main(_argv):
    writer = tf.io.TFRecordWriter(FLAGS.output_file)

    root_dir = pathlib.Path(FLAGS.data_dir).resolve()
    image_dir = root_dir / FLAGS.split

    images = sorted(list(image_dir.glob("*.png")))
    annotations = sorted(list(image_dir.glob("*.json")))

    for image, annotation in zip(tqdm.tqdm(images), annotations):
        record = build_record(image, annotation)
        writer.write(record.SerializeToString())

    writer.close()
    logging.info("Done")


if __name__ == "__main__":
    app.run(main)
