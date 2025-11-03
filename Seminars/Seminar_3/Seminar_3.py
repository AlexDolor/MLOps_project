import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.profiler as profiler
from torch.quantization import quantize_dynamic
import onnxruntime as ort
import onnx
# from onnx import optimizer
import onnxoptimizer as onnx_optim
import time
import os
import pandas as pd

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

def get_total_time(profiler):
    return sum([item.cpu_time_total for item in profiler.key_averages()])

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
    diff_max = differences.max()
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
    # print(f'{res1.shape=}')
    # print(f'{res2.shape=}')
    # print(f'{res1=}\n{res2=}')
    diff = np.linalg.norm(res1 - res2, axis=1)
    # print(f'{diff=}')
    # print(f'res1_norm={np.linalg.norm(res1, axis=1)}')
    rel_dif = diff/np.linalg.norm(res1, axis=1)
    diff_max = diff.max()
    rel_dif = rel_dif.max()
    print(f'relative {rel_dif}')
    print(f'absolute {diff_max}')
    return rel_dif, diff_max

def measure_performance(sess, loader, n=10):
    outputs = []
    labels = []
    start = time.time()
    for i, data in tqdm(enumerate(loader)):
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


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        # lambda x: (x * 2) - 1,
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
    save_path = 'Seminars/Seminar_3/'
    # for img, label in testloader:
    #     break
    # print(net(img))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    ## train(net, trainloader, criterion, optimizer)
    ## eval(net, testloader, criterion)
    # net = train_model(net, trainloader, testloader, criterion, optimizer, n_epochs=5)
    # torch.save(net.state_dict(), save_path + 'weights/MnistNet.pt')
    net.load_state_dict(torch.load(save_path + 'weights/MnistNet.pt', weights_only=True))
    for img, label in testloader:
        break
    
    # print('Translating model begins')
    # for img, label in testloader:
    #     break
    # traced = torch.jit.trace(net, img)
    # traced.save(save_path + "model_traced.pt")
    
    # scripted = torch.jit.script(net)
    # scripted.save(save_path + "model_scripted.pt")

    # for i, data in enumerate(testloader):
    #     img, label = data
    #     with torch.no_grad():
    #         org = net(img)
    #         jit = scripted(img)
    #     diff = torch.linalg.vector_norm(org - jit, dim=1)
    #     res = diff/torch.linalg.vector_norm(org, dim=1)
    #     print(f'relative {res.max()}')
    #     print(f'absolute {diff.abs().max()}')
    #     print(org[0])
    #     print(jit[0])
    #     break

    timings = {}

    # Проверка FP16
    # for img, label in testloader:
    #     break
    # with profiler.profile(record_shapes=True) as prof:
    #     with profiler.record_function("model_inference_fp32"):
    #         eval(net, testloader, criterion)
    #         # print('lol')
    # print(prof.key_averages().table(sort_by="cpu_time_total"))
    # assert 1==2
    # timings['default_model_time'] = get_total_time(prof)
    # torch.cpu.synchronize()
    # with profiler.profile(record_shapes=True) as prof:
    #     with profiler.record_function("model_inference_fp16"):
    #         with torch.autocast(device_type='cpu', dtype=torch.float16):
    #             eval(net, testloader, criterion)
    # torch.cpu.synchronize()
    # print(prof.key_averages().table(sort_by="cpu_time_total"))
    # timings['FP16_model_time'] = prof.self_cpu_time_total
    ##fp16 даже медленнее...

    ##Квантование модели
    # model_int8 = quantize_dynamic(net, {torch.nn.Linear}, dtype=torch.qint8)
    # with profiler.profile(record_shapes=True) as prof:
    #     with profiler.record_function("model_inference"):
    #         eval(model_int8, testloader, criterion)
    # timings['quant_model_time'] = get_total_time(prof)

    # print(timings)
    # models_differences(net, model_int8, testloader)
    # torch.save(model_int8.state_dict(), save_path + 'weights/MnistNet_quant.pt')

    ## Экспортируем модель
    # torch.onnx.export(
    #     net, img, save_path+"model.onnx",
    #     input_names=["input"], output_names=["output"],
    #     dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    #     opset_version=13
    # )

    # session = ort.InferenceSession(save_path+"model.onnx")

    ## Готовим входы: example_input надо преобразовать к numpy (и, если требуется, снять с GPU)
    # example_np = img.cpu().numpy() if hasattr(img, "cpu") else img

    # Если батч — используйте shape (N, C, H, W). Для одиночного, возможно, добавьте batch dim: example_np = example_np[None]

    # Запускаем инференс
    # outputs = session.run(
    #     output_names=["output"],   # если не указать — вернёт список всех выходов
    #     input_feed={"input": example_np}
    # )
    # print(type(outputs)==type([]))
    # print(outputs)
    # with torch.no_grad():
    #     result_differences(net(img), outputs[0])

    ##optimize model
    # passes = ["eliminate_deadend", "fuse_add_bias_into_conv",
    # "fuse_bn_into_conv"]
    # model_optimized = onnx_optim.optimize(onnx.load(save_path+"model.onnx"), passes)
    # onnx.save(model_optimized, save_path+"model_opt.onnx")

    # see the difference between optimized and unoptimized
    sess_og = ort.InferenceSession(save_path + 'model.onnx')
    sess_opt = ort.InferenceSession(save_path + 'model_opt.onnx')

    lat_og, throgh_og, acc_og, dict_og = measure_performance(sess_og, testloader, 10000)
    lat_opt, throgh_opt, acc_opt, _ = measure_performance(sess_opt, testloader, 10000)

    # print(f'outputs {dict_og["outputs"][0]}')
    # print(f'labels {dict_og["labels"][0]}')

    size_og = os.path.getsize(save_path + 'model.onnx') / 1024 / 1024
    size_opt = os.path.getsize(save_path + 'model_opt.onnx') / 1024 / 1024

    metrics_df = pd.DataFrame(columns=['model', 'size (Mb)', 'latency (ms)', 'throughput (obj/sec)', 'accuracy'])
    metrics_df.loc[0] = ['original', size_og, lat_og, throgh_og, acc_og]
    metrics_df.loc[1] = ['optimized', size_opt, lat_opt, throgh_opt, acc_opt]
    print(metrics_df)