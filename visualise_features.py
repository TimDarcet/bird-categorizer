import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import math


class AnnoteFinder(object):
    """callback for matplotlib to display an annotation when points are
    clicked on.  The point which is closest to the click and within
    xtol and ytol is identified.
    
    Register this function like this:
    
    scatter(xdata, ydata)
    af = AnnoteFinder(xdata, ydata, annotes)
    connect('button_press_event', af)
    """

    def __init__(self, xdata, ydata, annotes, ax=None, xtol=None, ytol=None):
        self.data = list(zip(xdata, ydata, annotes))
        if xtol is None:
            xtol = ((max(xdata) - min(xdata))/10.0)/2
        if ytol is None:
            ytol = ((max(ydata) - min(ydata))/10.0)/2
        self.xtol = xtol
        self.ytol = ytol
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        self.drawnAnnotations = {}
        self.links = []

    def distance(self, x1, x2, y1, y2):
        """
        return the distance between two points
        """
        return(math.sqrt((x1 - x2)**2 + (y1 - y2)**2))

    def __call__(self, event):
        if event.inaxes:
            clickX = event.xdata
            clickY = event.ydata
            if (self.ax is None) or (self.ax is event.inaxes):
                annotes = []
                # print(event.xdata, event.ydata)
                for x, y, a in self.data:
                    # print(x, y, a)
                    if ((clickX-self.xtol < x < clickX+self.xtol) and
                            (clickY-self.ytol < y < clickY+self.ytol)):
                        annotes.append(
                            (self.distance(x, clickX, y, clickY), x, y, a))
                if annotes:
                    annotes.sort()
                    distance, x, y, annote = annotes[0]
                    self.drawAnnote(event.inaxes, x, y, annote)
                    for l in self.links:
                        l.drawSpecificAnnote(annote)

    def drawAnnote(self, ax, x, y, annote):
        """
        Draw the annotation on the plot
        """
        if (x, y) in self.drawnAnnotations:
            markers = self.drawnAnnotations[(x, y)]
            for m in markers:
                m.set_visible(not m.get_visible())
            self.ax.figure.canvas.draw_idle()
        else:
            t = ax.text(x, y, " - %s" % (annote),)
            m = ax.scatter([x], [y], marker='d', c='r', zorder=100)
            self.drawnAnnotations[(x, y)] = (t, m)
            self.ax.figure.canvas.draw_idle()

    def drawSpecificAnnote(self, annote):
        annotesToDraw = [(x, y, a) for x, y, a in self.data if a == annote]
        for x, y, a in annotesToDraw:
            self.drawAnnote(self.ax, x, y, a)

# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 feature visualisation script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

# Data initialization and loading
from data import eval_transforms
def valid(fn):
    return fn.split('.')[-2] == '00'

train_ds = datasets.ImageFolder(args.data + '/train_images',
                                transform=eval_transforms,
                                is_valid_file=valid)

train_loader = torch.utils.data.DataLoader(train_ds,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=1)

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from featurizer import Net
model = Net()
if args.model is not None:
    state_dict = torch.load(args.model)
    model.load_state_dict(state_dict)
model.eval()
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

outputs = np.empty((0, 2048))
labels = np.empty((0))
model.eval()
for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
    if use_cuda:
        data, target = data.cuda(), target.cuda()
    output = model(data).cpu().detach().numpy()
    outputs = np.concatenate((outputs, output), axis=0)
    labels = np.concatenate((labels, target.cpu().numpy()), axis=0)

lab_names = [train_loader.dataset.classes[int(l)] for l in labels]
imgs_fn = [i[0] for i in train_loader.dataset.imgs]

pca = PCA(n_components=2)
projectedPCA = pca.fit_transform(outputs)
tsne = TSNE(n_components=2)
projectedTSNE = tsne.fit_transform(outputs)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(projectedPCA[:, 0], projectedPCA[:, 1], c=labels)
ax2.scatter(projectedTSNE[:, 0], projectedTSNE[:, 1], c=labels)

pcaaf = AnnoteFinder(projectedPCA[:, 0], projectedPCA[:, 1], imgs_fn)
tsneaf = AnnoteFinder(projectedTSNE[:, 0], projectedTSNE[:, 1], imgs_fn)
def on_enter_event(_):
    fig.canvas.setFocus()
fig.canvas.mpl_connect('button_press_event', pcaaf)
fig.canvas.mpl_connect('button_press_event', tsneaf)
fig.canvas.mpl_connect('axes_enter_event', on_enter_event)

fig.show()

