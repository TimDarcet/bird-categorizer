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
        self.model_list = [
            "vgg16",
            "resnet50",
            "resnet152",
            # "squeezenet1_0",
            # "squeezenet1_1",
            "densenet121",
            # "inception_v3",
            "googlenet",
        ]
        self.featurizers = nn.ModuleList([models.__dict__[m_name](pretrained=True) for m_name in self.model_list])
        for feater in self.featurizers:
            for param in feater.parameters():
                param.requires_grad = False
        self.n_features = []
        for mn, feater in zip(self.model_list, self.featurizers):
            if mn[:3] == "vgg":
                self.n_features.append(feater.classifier[0].in_features)
                feater.classifier = Identity()
            elif mn[:6] == "resnet":
                self.n_features.append(feater.fc.in_features)
                feater.fc = Identity()
            elif mn[:10] == "squeezenet":  # MEF: wants 227x227
                self.n_features.append(512)
                feater.classifier = nn.AdaptiveAvgPool2d((1, 1))
            elif mn[:8] == "densenet":
                self.n_features.append(feater.classifier.in_features)
                feater.classifier = Identity()
            elif mn == "inception_v3":  # MEF: wants 299x299
                self.n_features.append(feater.fc.in_features)
                feater.fc = Identity()
            elif mn == "googlenet":
                self.n_features.append(feater.fc.in_features)
                feater.fc = Identity()
        self.dp = nn.Dropout(0.7)
        self.classifier = nn.Linear(sum(self.n_features), nclasses)

    def forward(self, x):
        featlist = tuple(feater(x) for feater in self.featurizers)
        features = torch.cat(featlist, dim=1)
        drops = self.dp(features)
        return self.classifier(drops)
