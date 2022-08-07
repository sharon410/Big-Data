import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.datasets as datasets

start_time=time.time()

class Model(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Model, self).__init__()
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

def test_accuracy(model, data_loader):
    print("Test started...")
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in data_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images : %d %%' % (100 * correct / total))


def train_model(model, data_loader):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss()

    for data, labels in data_loader:
        optimizer.zero_grad()
        loss = criterion(model(data), labels)
        loss.backward()        
        optimizer.step()


if __name__ == '__main__':    
    num_processes = 4
    model = Model()
    model.share_memory()
    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))]))
    testset = datasets.MNIST('./data', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))]))
    processes = []
    for rank in range(num_processes):
        data_loader = DataLoader(
            dataset=dataset,
            sampler=DistributedSampler(
                dataset=dataset,
                num_replicas=num_processes,
                rank=rank
            ),
            batch_size=32
        )     
        p = mp.Process(target=train_model, args=(model, data_loader))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    test_accuracy(model, DataLoader(
        dataset=testset,
        batch_size=1000)) 
    print("\n Time taken for {} processes is {}".format(num_processes,round(time.time()-start_time,4)))