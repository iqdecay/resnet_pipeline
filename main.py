import logging

import torch
from GPUtil import showUtilization as gpu_usage

from pipeline import Pipeline
from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

if __name__ == '__main__':
    res18 = (ResNet18(), "18")
    res34 = (ResNet34(), "34")
    res50 = (ResNet50(), "50")
    res101 = (ResNet101(), "101")
    all_resnets = [res18, res34, res50, res101]
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S'
                        )

    for model, architecture in all_resnets:
        for n_epoch in [10, 50]:
            for lr in [0.1, 0.01, 0.001]:
                name = f"ResNet{architecture}_{lr}_{n_epoch}epoch"
                pipeline = Pipeline(model, lr, n_epoch, name)
                pipeline.run()
