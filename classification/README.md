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

```sh

```

## Evaluation


```sh
# You can download pretrained model here 
wget https://github.com/ryanking13/kcaptcha-solver/releases/download/v0.1/l2.model.h5

python train.py --eval-only -o l2.model.h5
```

```sh
tensorflowjs_converter --input_format keras --output_format=tfjs_graph_model model.h5 model_tfjs/
```