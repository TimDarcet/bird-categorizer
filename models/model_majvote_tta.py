import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models.model_majvote import Net as InnerNet
import ttach



nclasses = 20 


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.transforms = ttach.Compose(
            [
                ttach.HorizontalFlip(),
                ttach.Scale(scales=[1, 1.05], interpolation="linear"),
                ttach.Multiply(factors=[0.95, 1, 1.05]),
            ]
        )
        self.model = ttach.ClassificationTTAWrapper(InnerNet(),
                                                    transforms=self.transforms,
                                                    merge_mode="mean")
    def forward(self, x):
        return self.model(x)
