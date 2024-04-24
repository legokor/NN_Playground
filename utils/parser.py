import torch.nn as nn
from torchvision import transforms
import lightning as L
import yaml

class Parser:
    def read_config(config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def network(layers):
        modules = []
        for layer in layers:
            modules.append(getattr(nn, layer[2])(*layer[3]))
        return nn.Sequential(*modules)

    def transforms(transform_list):
        transforms_list = []
        for transform in transform_list:
            transform_name = transform[0]
            transform_args = transform[1]
            transform = getattr(transforms, transform_name)(*transform_args)
            transforms_list.append(transform)
        return transforms.Compose(transforms_list)

    def callbacks(callbacks_list):
        callbacks = []
        for callback in callbacks_list:
            callback_name = callback[0]
            callback_args = callback[1:]
            callback = getattr(L.pytorch.callbacks, callback_name)(callback_args)
            callbacks.append(callback)
        return callbacks

    def loss_fun(loss_function):
        return getattr(nn.functional, loss_function)
