import zipfile
import os

import torchvision.transforms as transforms
from PIL.Image import BICUBIC

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224), interpolation=BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.05),
    # transforms.ToTensor(),
    # transforms.ToPILImage(),
    transforms.Resize((224, 224), interpolation=BICUBIC),
    transforms.RandomAffine(degrees=15, translate=[0.1, 0.1], scale=[0.9, 1.1], shear=[-10, 10], resample=BICUBIC),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2))], p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# train_transforms = eval_transforms


