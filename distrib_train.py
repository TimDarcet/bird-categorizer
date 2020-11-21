import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, random_split, ChainDataset, DistributedSampler
from torchvision import datasets
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from data import train_transforms
from model import Net


def init_parser():
    dataset_default = 'crops_square/bird_dataset'
    epochs_default = 20
    batch_size_default = 2048
    lr_default = 0.01
    momentum_default = 0.5
    log_interval_default = 10
    experiment_default = 'experiment'
    train_val_prop_default = 0.8
    random_seed_default = 42
    optimizer_default = "SGD"
    parser = argparse.ArgumentParser(description='RecVis A3 training script')
    parser.add_argument("--local_rank",
                        type=int,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument('--data',
                        type=str,
                        default=dataset_default,
                        metavar='D',
                        help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
    parser.add_argument('--batch-size',
                        type=int,
                        default=batch_size_default,
                        metavar='B',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs',
                        type=int,
                        default=epochs_default,
                        metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr',
                        type=float,
                        default=lr_default,
                        metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum',
                        type=float,
                        default=momentum_default,
                        metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed',
                        type=int,
                        default=random_seed_default,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval',
                        type=int,
                        default=log_interval_default,
                        metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--experiment',
                        type=str,
                        default=experiment_default,
                        metavar='E',
                        help='folder where experiment outputs are located.')
    parser.add_argument('--train-val-prop',
                        type=float,
                        default=train_val_prop_default,
                        metavar='TVP',
                        help='proportion of images to use for train set.')
    parser.add_argument("--random-seed",
                        type=int,
                        help="Random seed.",
                        default=random_seed_default)
    parser.add_argument("--optimizer",
                        type=str,
                        help="Optimizer (adam / SGD).",
                        default=optimizer_default)
    return parser.parse_args()


def set_random_seeds(random_seed=42):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(data_folder, tv_prop, batch_size):
    full_ds = datasets.ImageFolder(data_folder + '/train_val_images',
                                   transform=train_transforms)
    train_size = int(tv_prop * len(full_ds))
    val_size = len(full_ds) - train_size
    print(f"Using train size {train_size} and val size {val_size} with batch size {batch_size}")
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    train_sampler = DistributedSampler(train_ds)
    val_sampler = DistributedSampler(val_ds)
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=0)
                                            #    drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds,
                                             batch_size=batch_size,
                                             sampler=val_sampler,
                                             num_workers=0)
    return train_loader, val_loader


def train(epoch, model, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    return loss.cpu().item()


def validation(model, val_loader, optimizer, criterion):
    model.eval()
    validation_loss = torch.Tensor([0]).to("cuda")
    correct = torch.Tensor([0]).to("cuda")
    for data, target in val_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        # batch loss
        validation_loss += criterion(output, target)
        # batch accuracy
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).int().sum()
    return validation_loss.cpu().item() / len(val_loader), correct.cpu().item()


def main():
    # Parse arguments
    args = init_parser()

    # We need to use seeds to make sure that the models initialized in different processes are the same
    set_random_seeds(args.random_seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend="gloo")  # TODO: try NCCL backend

    # Data initialization and loading
    train_loader, val_loader = load_data(args.data,
                                         args.train_val_prop,
                                         args.batch_size)

    # Define the model 
    model = Net()
    model.cuda()

    # Distribute the model
    model = DistributedDataParallel(model,
                                    device_ids=[args.local_rank],
                                    output_device=args.local_rank)
    
    # Define optimizer
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError()

    # Define loss
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # Run the training
    for epoch in range(1, args.epochs + 1):
        last_train_loss = train(epoch, model, train_loader, optimizer, criterion)
        avg_val_loss, val_corrects = validation(model, val_loader, optimizer, criterion)
        val_acc = val_corrects / len(val_loader.dataset)
        print(f"Epoch {epoch:02d}: last train loss = {last_train_loss:03f} | avg val loss = {avg_val_loss:03f} | val acc = {val_acc:03f} ({int(val_corrects):03d}/{len(val_loader.dataset)})")
        model_file = args.experiment + '/model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        # print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')

if __name__ == "__main__":
    main()
