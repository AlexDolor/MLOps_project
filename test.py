# import torch
# import numpy
# x = torch.rand(5, 3)
# print(x)
# print('Hello World!')
import torch
# import torchvision
# import matplotlib
# import onnxruntime
# import tensorboard
# print("Все библиотеки успешно установлены!")

# x = torch.tensor(range(10),  dtype=float)
# y = torch.tensor(range(10,0,-1),  dtype=float)
# # print(torch.sqrt(torch.sum(x**2)))
# # print(torch.linalg.vector_norm(x))
# x = x.view(2, -1)
# y = y.view(2, -1)
# print(x)
# print(x - y)
# print(torch.linalg.vector_norm(x, dim=1))

# x = [1,2,3]
# y = [5,6]
# print(x+y)

import numpy as np
# x = [1,-2,3]
# print(np.abs(x).max())
# x = x.numpy()
# y = y.numpy()
# print(x)
# print(np.linalg.norm(x, axis=1))
# print(x-y)
# print(np.linalg.norm(x-y, axis=1))
# z = np.concat([x,x])
# print(z)
# print(z.shape)
# print(np.max(z, 1))
# print(np.argmax(z, 1))

import torchvision.transforms as transforms
transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
        ])
img = torch.randint(0, 256, size=(4, 3, 28, 28), dtype=torch.uint8).numpy()
print(img.shape)
print(transform(img).shape)