import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, random_split, ChainDataset
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='crops_square/bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--train-val-prop', type=float, default=0.9, metavar='TVP',
                    help='proportion of images to use for train set.')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading
from data import train_transforms
def valid(fn):
    return fn.split('.')[-2] == '00' or True


# train_f_ds = datasets.ImageFolder(args.data + '/train_images',
#                                   transform=train_transforms,
#                                   is_valid_file=valid)
# val_f_ds = datasets.ImageFolder(args.data + '/val_images',
#                                 transform=train_transforms,
#                                 is_valid_file=valid)
full_ds = datasets.ImageFolder(args.data + '/train_val_images',
                                transform=train_transforms)
train_size = int(args.train_val_prop * len(full_ds))
val_size = len(full_ds) - train_size
print(f"Using train size {train_size} and val size {val_size}")
# full_ds = ConcatDataset([train_f_ds, val_f_ds])
# train_size = int(args.train_val_prop * len(full_ds))
# val_size = len(full_ds) - train_size
# print(f"Using train size {train_size} and val size {val_size}")
train_ds, val_ds = random_split(full_ds, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net
model = Net()
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
# optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.cpu().item()))


def validation():
    model.eval()
    validation_loss = torch.Tensor([0]).to("cuda")
    correct = torch.Tensor([0]).to("cuda")
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        validation_loss += criterion(output, target)
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]

        correct += pred.eq(target.data.view_as(pred)).int().sum()
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss.data.item() / len(val_loader), correct.data.item(), len(val_loader.dataset),
        100. * correct.data.item() / len(val_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation()
    model_file = args.experiment + '/model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
