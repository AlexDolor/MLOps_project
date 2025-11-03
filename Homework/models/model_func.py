import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time

class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        # rough formula for img size after conv
        # (size - kernel + 2padding)/stride + 1
        #1*28*28 -> 4*28*28 
        self.conv0 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        #4*28*28 -> 8*14*14 
        self.conv1 = nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1)
        #8*14*14 -> 16*7*7
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
        #16*7*7 -> 32*4*4
        # (7 - 3 + 2)/2 +1
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        
        self.fc1 = nn.Linear(32*4*4, 64)
        self.fc2 = nn.Linear(64, 10)
        
        #doesnt change size
        self.avgpool_mid = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # activation func
        self.activ = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=0.4)
    
    def forward(self, x):
        x = self.activ(self.conv0(x))
        x = self.maxpool(x)
        x = self.activ(self.conv1(x))
        x = self.avgpool_mid(x)
        x = self.activ(self.conv2(x))
        x = self.maxpool(x)
        x = self.activ(self.conv3(x))
        x = self.avgpool_mid(x)
        x = x.view(x.size(0), -1)
        x = self.activ(self.fc1(x))
        x = self.dropout(x)
        x = self.activ(self.fc2(x))
        return x

def train(model, trainloader, criterion, optimizer):
    losses = []
    for inputs, labels in tqdm(trainloader):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
    return sum(losses) / len(losses)

def eval(model, testloader, criterion):
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
    return sum(losses) / len(losses)

def print_losses(train_losses, test_losses):
    plt.plot(train_losses, label='train')
    plt.plot(test_losses, label='test')
    plt.legend()
    plt.title('MSE Loss')
    plt.show()
    return 

def train_model(model, trainloader, testloader, criterion, optimizer, n_epochs=3):
    train_losses = []
    test_losses = []
    for epoch in tqdm(range(n_epochs)):
        train_loss = train(model, trainloader, criterion, optimizer)
        test_loss = eval(model, testloader, criterion)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    print_losses(train_losses, test_losses)
    return model   

def models_differences(model1, model2, loader, n=10):
    differences = []
    absolute_differences = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            img, label = data    
            out1 = model1(img)
            out2 = model2(img)
            diff = torch.linalg.vector_norm(out1 - out2, dim=1)
            rel = diff/torch.linalg.vector_norm(out1, dim=1)
            differences += rel
            absolute_differences += diff
            if i >= n: break
    diff_max = max(differences)
    abs_diff_max = max(absolute_differences)
    print(f'relative {diff_max}')
    print(f'absolute {abs_diff_max}')
    return diff_max, abs_diff_max 


def result_differences(res1, res2):
    '''for onnx
    res1 :torch.tensor:
    res2 :list:'''
    res1 = res1.numpy()
    res2 = np.array(res2)
    diff = np.linalg.norm(res1 - res2, axis=1)
    rel_dif = diff/np.linalg.norm(res1, axis=1)
    diff_max = diff.max()
    rel_dif = rel_dif.max()
    print(f'relative {rel_dif}')
    print(f'absolute {diff_max}')
    return rel_dif, diff_max

def measure_performance_onnx(sess, loader, n=10):
    '''for onnx models'''
    outputs = []
    labels = []
    start = time.time()
    for i, data in enumerate(tqdm(loader)):
        img, label = data
        img = img.numpy()
        label = label.numpy()

        outputs.append(sess.run(output_names=["output"], input_feed={"input": img})[0])
        labels.append(label)
        if i>=n: break
    total_time = time.time() - start
    # calc accuracy
    outputs = np.concat(outputs)
    labels = np.concat(labels)
    obj_number = outputs.shape[0]
    predicted = np.argmax(outputs, 1)
    correct = (predicted == labels).sum()
    accuracy = correct / (obj_number)
    latency = (total_time / obj_number) * 1000  # ms per request
    throughput = obj_number / total_time
    return latency, throughput, accuracy, {'outputs':outputs, 'labels':labels}

def measure_performance_torch(model, loader, n=10):
    '''for torch models.
    returns
    latency, throughput, accuracy, dict with outputs and labels }'''
    outputs = []
    labels = []
    model.eval()
    with torch.no_grad():
        start = time.time()
        for i, data in enumerate(tqdm(loader)):
            img, label = data
            img = img
            label = label

            outputs.append(model(img).numpy())
            labels.append(label.numpy())
            if i>=n: break
        total_time = time.time() - start
    # calc accuracy
    outputs = np.concat(outputs)
    labels = np.concat(labels)
    obj_number = outputs.shape[0]
    predicted = np.argmax(outputs, 1)
    correct = (predicted == labels).sum()
    accuracy = correct / (obj_number)
    latency = (total_time / obj_number) * 1000  # ms per request
    throughput = obj_number / total_time
    return latency, throughput, accuracy, {'outputs':outputs, 'labels':labels}