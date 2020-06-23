
# Global Wheat Detection competition solution by sashluyak

You can get more details on [Kaggle competition page](https://www.kaggle.com/c/global-wheat-detection)

### Create virtual env

```bash
poetry run pip install --upgrade pip
poetry install
```

### Download and extract data

```bash
poetry run kaggle competitions download -c global-wheat-detection -p ./data
unzip ./data/global-wheat-detection.zip -d ./data
```

EfficientDet implementation by [kamauz](https://github.com/kamauz/EfficientDet).


### Download pretrained Efficientnet weights

```bash
mkdir weights
wget -P ./weights https://github.com/Callidior/keras-applications/releases/download/efficientnet/efficientnet-b4_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5
```
