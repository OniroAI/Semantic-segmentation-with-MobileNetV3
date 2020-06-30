import os
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer

from modules.loss import FbCombinedLoss, BinaryCrossentropy
from modules.keras_models import MobileNetV3SmallSegmentation

# Define dictionaries with modules names
loss_dict = {'fb_combined': FbCombinedLoss,
             'bce': BinaryCrossentropy}

model_dict = {'mobilenet_small': MobileNetV3SmallSegmentation}


def get_optimizer(optimizer_name):
    optimizers_dict = {subcl.__name__: subcl
                       for subcl in Optimizer.__subclasses__()}
    assert optimizer_name in optimizers_dict.keys(
    ), "Optimizer name is not in PyTorch optimizers"
    return optimizers_dict[optimizer_name]


def get_model(model_name, **kwargs):
    assert model_name in model_dict.keys(), "Unknown model name"
    model = model_dict[model_name](**kwargs)
    return model


def get_loss(loss_name, **kwargs):
    assert loss_name in loss_dict.keys(), "Unknown loss name"
    loss_function = loss_dict[loss_name](**kwargs)
    return loss_function


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
