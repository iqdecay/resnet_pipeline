import logging
import os

import torch

from pipeline import Pipeline
from constants import MODEL_DIR
from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from report import Report

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S'
                        )
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    res18 = (ResNet18(), "18")
    res34 = (ResNet34(), "34")
    res50 = (ResNet50(), "50")
    res101 = (ResNet101(), "101")
    res152 = (ResNet101(), "152")
    all_resnets = [res18, res34, res50, res101, res152]

    for model, architecture in all_resnets:
        for lr in [0.1, 0.01, 0.001]:
            for n_epoch in [10, 50]:
                name = f"ResNet{architecture}_{lr}_{n_epoch}epoch"
                pipeline = Pipeline(model, lr, n_epoch, name)
                pipeline.run()
