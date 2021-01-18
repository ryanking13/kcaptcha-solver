# KCAPTCHA Solver object detection model

Model implementation fully adapted from [yolo3-tf2](https://github.com/zzh8829/yolov3-tf2).

## Requirements / Installation

- Python >= 3.6

```
pip install -r requirements.txt

# Convert pre-trained Darknet yolo V3 tiny weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights -O weights/yolov3-tiny.weights
python convert.py --weights ./weights/yolov3-tiny.weights --output ./checkpoints/yolov3-tiny.tf --tiny

# Check conversion is done correctly
python detect.py --weights ./checkpoints/yolov3-tiny.tf --tiny --image ./data/street.jpg
```

## Getting datasets

```
wget https://github.com/ryanking13/kcaptcha-generator/releases/download/v1.0/default.data.l2.10000.zip -o .data.zip
unzip .data.zip -o .data
```

> If you want to make your custom dataset, see https://github.com/ryanking13/kcaptcha-generator

## Preprocessing datasets

Datasets need to be converted to TFRecord format.

```
python tools/kcaptcha.py --data_dir .data --split train --output_file .data/kcaptcha_train.tfrecord
python tools/kcaptcha.py --data_dir .data --split validation --output_file .data/kcaptcha_validation.tfrecord
python tools/kcaptcha.py --data_dir .data --split test --output_file .data/kcaptcha_test.tfrecord
```


## Training

```sh
python train.py -v
```

```
usage:
```

## Evaluation


```sh
# You can download pretrained model here 
```

## Converting model weights to tf.js

```sh
```

TODO

- check tfrecord visualization
- training
- conversion
- clean up repo