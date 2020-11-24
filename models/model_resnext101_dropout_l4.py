import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


nclasses = 20 

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.featurizer = torch.hub.load('pytorch/vision:v0.6.0', 'resnext101_32x8d', pretrained=True)
        num_ftrs = self.featurizer.fc.in_features
        for param in self.featurizer.parameters():
            param.requires_grad = False
        for param in self.featurizer.layer4.parameters():
            param.requires_grad = True
        self.featurizer.fc = Identity()
        self.dp = nn.Dropout(0.6)
        self.classifier = nn.Linear(num_ftrs, nclasses)

    def forward(self, x):
        feats = self.featurizer(x)
        drops = self.dp(feats)
        return self.classifier(drops)
