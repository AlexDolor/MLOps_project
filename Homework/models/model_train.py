import torchvision
import torchvision.transforms as transforms
# import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from model_func import MnistNet, train, eval, train_model

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
        ])
    batch_size = 4
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    
    net = MnistNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net = train_model(net, trainloader, testloader, criterion, optimizer, n_epochs=5)
    torch.save(net.state_dict(), 'weights/MnistNet.pt')
    
    
