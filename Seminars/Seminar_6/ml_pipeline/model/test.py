import onnxruntime as ort, numpy as np


import torchvision
import torchvision.transforms as transforms
import torch

if __name__ == '__main__':
    transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
            ])
    batch_size = 4
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
    for img, label in testloader:
        img = img.numpy()
        break
    session = ort.InferenceSession("model.onnx")
    x = np.array(img.tolist(), dtype=np.float32)#[None, ...]
    print(x.shape)
    output = session.run(None, {"input": x})[0]
    print(output)