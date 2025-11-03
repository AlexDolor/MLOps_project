import requests
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np

if __name__ == '__main__':
    # test = 'model'
    test = 'preprocess'
    # test = 'postprocess'

    match test:
        case 'preprocess':
            # test preprocess
            image_path = "Seminars\Seminar_6\image.png"

            with open(image_path, "rb") as img_file:
                files = {"file": img_file}
                response = requests.post("http://localhost:5001/process", files=files)

            print(response.json())

        case 'model':
            #test model
            transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))
                    ])
            batch_size = 4
            testset = torchvision.datasets.MNIST(root='./data', train=False,
                                                    download=True, transform=transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                        shuffle=False, num_workers=2)
            for img, label in testloader:
                img = img.numpy()
                break
            tensor = img # возьми из ответа preprocess
            # print(img)
            r = requests.post('http://localhost:5002/predict', json={"tensor": tensor.tolist()})
            print(r.json())

        case 'postprocess':
            # test postprocess
            probabilities = [[0,1,2,3,4,5,6,7,8,9]] # {'result_class': [9]} it works!!!!
            r = requests.post('http://localhost:5003/postprocess', json={"result": probabilities})
            print(r.json())


