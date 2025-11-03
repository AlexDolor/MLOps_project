import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
# Задание 1. Работа с тензорами
# Создайте два тензора: один с равномерным распределением значений, другой — с нормальным.
# Выполните между ними элементные арифметические операции: сложение, умножение.
# Преобразуйте результат в массив NumPy и выведите его.

# print('\nЗадание 1\n')
# shape = (2,3)
# a = torch.rand(shape)
# b = torch.randn(shape)
# print(f'a + b = \n{(a+b).numpy()}')
# print(f'a * b = \n{(a*b).numpy()}')

# Задание 2. Индексация и манипуляции с тензорами
# Создайте тензор размера  4×4, заполненный случайными числами.
# Получите центральный 2×2 блок, выделив его срезами.
# Измените значения элементов в этом блоке на 1.

# print('\nЗадание 2\n')
# torch.seed = 42
# a = torch.rand(4,4)
# print(a[1:3,1:3])
# print(a)
# a[1:3,1:3] += 1
# print(a[1:3,1:3])
# print(a)


# Задание 3. Работа с градиентами
# Создайте тензор x=3.0 с requires_grad=True.
# Определите функцию y=5x^3−2x^2+x+1.
# Вычислите градиент функции y в точке x=3.0.

# print('\nЗадание 3\n')
# x = torch.tensor(3.0, requires_grad=True)
# y = 5*x**3 - 2*x**2 + x + 1
# y.backward()
# print(x.grad)

# Задание 4. Создание собственной нейронной сети
# Реализуйте класс нейронной сети с двумя полносвязными слоями (nn.Linear) и активацией ReLU.
# Задайте входное и выходное количество нейронов: 10 и 2 соответственно.
# Проверьте, что модель возвращает выходной тензор нужной размерности, передав случайный вход.

# print('\nЗадание 4\n')
# class PersonalNN(nn.Module):
#     def __init__(self):
#         super(PersonalNN, self).__init__()
#         self.fc1 = nn.Linear(10, 5)
#         self.fc2 = nn.Linear(5, 2)
    
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return x
    
# pers_nn = PersonalNN()
# result = pers_nn.forward(torch.rand(1,10)) 
# print(result.shape)

# Задание 5. Обучение нейронной сети на наборе данных MNIST
# Загрузите данные MNIST с использованием torchvision.datasets.
# Нормализуйте данные и создайте DataLoader.
# Обучите простую нейронную сеть с двумя линейными слоями (используйте ReLU и CrossEntropyLoss) на этих данных.


class SimpleCVNN(nn.Module):
    def __init__(self):
        super(SimpleCVNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 60)
        self.fc2 = nn.Linear(60 , 10)
    
    def forward(self, x):
        # batch_size = x.size(0)
        # x = self.fc(x).view(batch_size, 1, 28, 28)
        x = x.view(-1,28*28)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class NotSimpleSimpleCVNN(SimpleCVNN):
    def __init__(self):
        super(NotSimpleSimpleCVNN, self).__init__()
        self.dropout = nn.Dropout(p=0.2)   
    
    def forward(self, x):
        # batch_size = x.size(0)
        # x = self.fc(x).view(batch_size, 1, 28, 28)
        x = x.view(-1,28*28)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
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

if __name__ == '__main__':
    print('\nЗадание 5\n')


    ## transformations
    transform = transforms.Compose(
        [transforms.ToTensor(),
        # lambda x: (x * 2) - 1,
        transforms.Normalize((0.5), (0.5))
        ])
    batch_size = 4
    ## torchvision.datasets #MNIST
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)


    net = SimpleCVNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    save_path = 'Seminars/Seminar_1/'
    n_epochs = 3

    # for img, lab in trainloader:
    #     print(img.shape)
    #     break
    # print(net(img))

    ## train and save weights
    # train(net, trainloader, criterion, optimizer)
    # eval(net, testloader, criterion)
    # net = train_model(net, trainloader, testloader, criterion, optimizer, n_epochs=3)
    # torch.save(net.state_dict(), save_path + 'weights/simple.pt')
    net.load_state_dict(torch.load(save_path + 'weights/simple.pt', weights_only=True))
    newnet = NotSimpleSimpleCVNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(newnet.parameters(), lr=0.001, momentum=0.9)
    ## train and save weights
    # train(newnet, trainloader, criterion, optimizer)
    # eval(newnet, testloader, criterion)
    # newnet = train_model(newnet, trainloader, testloader, criterion, optimizer, n_epochs=3)
    # torch.save(newnet.state_dict(), save_path + 'weights/notsimple.pt')
    
    ## see the difference
    newnet.load_state_dict(torch.load(save_path + 'weights/notsimple.pt', weights_only=True))
    simple_loss = eval(net, testloader, criterion)
    notsimple_loss = eval(newnet, testloader, criterion)
    print(f'simplenet Loss {simple_loss}\nNotsimplenet Loss {notsimple_loss}')


# # Задание 6. Добавление Dropout
# # Расширьте модель из задания 5, добавив слой Dropout между слоями.
# # Обучите новую модель и сравните точность на тестовом наборе с базовой моделью.

# print('\nЗадание 5\n')

# class UpgradedCNNN(SimpleCVNN):
#     def __init__(self):
#         super(UpgradedCNNN, self).__init__()
#         self.dropout = nn.Dropout(0.2)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         return x
    

