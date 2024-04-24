[![](https://img.shields.io/badge/made%20by-Lego%20AI%20Team-orange.svg)](https://legokor.hu/projects/mesterseges_intelligencia/)
# General Neural Network training platform for classification or regression

This projects represents a basic playground for deep neural network training for learning purposes of the [LEGO AI Team](https://legokor.hu/projects/mesterseges_intelligencia/). It aims to use modern solutions (pytorch, lightning) with easy to use on a flexible platform.

## Install

```bash
git clone repo
cd repo
conda create -n nnet python=3.10
pip install -r requirements.txt
```

## Usage

The [nn folder](./nn) contains the model config files, and a basic implementation of a deep neural network.
The [scripts](./scripts) contains the config of the training methods (the data augmentations, epoch, early stopping, and so). It will have separate scripts for training, testing, etc.

The [tutorial.ipynb](tutorial.ipynb) contains a demo of usage.

## Contribution

Feel free to contribute, report issues or request features!