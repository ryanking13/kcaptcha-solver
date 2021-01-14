# KCAPTCHA Solver classification model

## Requirements / Installation

- Python >= 3.6

```
pip install -r requirements.txt
```

## Getting datasets

```
wget https://github.com/ryanking13/kcaptcha-generator/releases/download/v1.0/default.data.l2.10000.zip -o .data.zip
unzip .data.zip -o .data
```

> If you want to make your custom dataset, see https://github.com/ryanking13/kcaptcha-generator

## Training

```sh
python train.py -v
```

```
usage: train.py [-h] [-l LENGTH] [--width WIDTH] [--height HEIGHT] [--char-set CHAR_SET] [--train TRAIN] [--validation VALIDATION] [--test TEST]
                [--epochs EPOCHS] [--batch-size BATCH_SIZE] [-v] [-o OUTPUT] [--eval-only]

optional arguments:
  -h, --help            show this help message and exit
  -l LENGTH, --length LENGTH
                        Length of CAPTCHA (default: 2)
  --width WIDTH         Width of input (default: 160)
  --height HEIGHT       Height of input (default: 60)
  --char-set CHAR_SET   Available characters for CAPTCHA (default: 0123456789)
  --train TRAIN         Train dataset directory (default: .data/train)
  --validation VALIDATION
                        Validation dataset directory (default: .data/validation)
  --test TEST           Test dataset directory (default: .data/test)
  --epochs EPOCHS       Traning epochs (default: 5)
  --batch-size BATCH_SIZE
                        Batch size (default: 64)
  -v, --verbose
  -o OUTPUT, --output OUTPUT
                        Save best model to specified path, if not specified, model is not saved
  --eval-only           Evaluate trained model, must be used with --output option
```

## Evaluation


```sh
# You can download pretrained model here 
wget https://github.com/ryanking13/kcaptcha-solver/releases/download/v0.1/l2.model.h5

python train.py --eval-only -o l2.model.h5
```

## Converting model weights to tf.js

```sh
tensorflowjs_converter --input_format keras --output_format=tfjs_graph_model l2.model.h5 model_tfjs/
```