import pandas as pd
import os
from mpi4py import MPI
from PIL import Image
import time
import csv
import random
import numpy as np
import sys
import pdb
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F


gbatch_size = 128

from math import ceil
from random import Random

start= MPI.Wtime()


""" Dataset partitioning"""
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class Net(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(80, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(self.conv2_drop(F.max_pool2d(self.conv2(x), 2)))
        x=F.max_pool2d(x,2)
        x = x.view(-1, 80)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        
class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

    """ Partitioning MNIST """
def partition_dataset():
    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    size = comm.Get_size()
    bsz = 128 / float(size)
    bsz=int(bsz)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    rank = comm.Get_rank()
    partition = partition.use(rank)
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True)
    return train_set, bsz

""" Distributed Synchronous SGD"""
def run(rank, size):
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)
    num_batches = ceil(len(train_set.dataset) / float(bsz))
    rank = comm.Get_rank()       
    for epoch in range(10):
        epoch_loss = 0.0
        running_acc=0.0
        for data, target in train_set:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
            running_acc += accuracy(output,target)
    print('Loss Rank ', rank, ', epoch ', epoch, ': ', epoch_loss / num_batches)
    print('Accuracy Rank ', rank, ', epoch ', epoch, ': ', running_acc / num_batches)
    return model
    

def validate(test_loader,model):
    '''
    Function for the testing step of the training loop
    '''
    model.eval()
    running_loss = 0
    running_acc=0
    for X, y_true in test_loader:
        # Forward pass and record loss
        y_hat= model(X) 
        loss = F.nll_loss(y_hat, y_true)
        running_loss += loss.item() * X.size(0)
        running_acc += accuracy(y_hat, y_true)
    epoch_loss = running_loss / len(test_loader.dataset)

    return model, epoch_loss,running_acc

""" Gradient averaging. """
def average_gradients(model):
    size = comm.Get_size()
    for param in model.parameters():
        comm.allreduce(param.grad.data, op=MPI.SUM)
        param.grad.data /= size

def accuracy(output, y):
        pred_labels = torch.argmax(output, dim=1)
        return (pred_labels == y).sum().item() / len(y)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    model=run(rank, size)
    test_losses=[]
    test_dataset = datasets.MNIST('./data', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))

    
    test_loader = DataLoader(dataset=test_dataset, 
                                batch_size=64, 
                                shuffle=False)
    if rank==0:
        num_batches = ceil(len(test_loader.dataset) / float(64))
        for epoch in range(10):
            with torch.no_grad():
                model, test_loss,test_accuracy = validate(test_loader,model)
        print("Final Test Loss is",test_loss)
        print("Final Test Accuracy is",test_accuracy/num_batches)
        print("\n Time taken for {} workers = {}".format(size,round(MPI.Wtime()-start,4)))

