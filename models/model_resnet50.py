import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


nclasses = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.featurizer = models.resnet50(pretrained=True)
        num_ftrs = self.featurizer.fc.in_features
        for param in self.featurizer.parameters():
            param.requires_grad = False
        self.featurizer.fc = nn.Linear(num_ftrs, nclasses)

    def forward(self, x):
        return self.featurizer.forward(x)