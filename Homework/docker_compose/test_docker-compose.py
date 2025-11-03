import requests
# import torchvision
import torchvision.transforms as transforms
# import torch
# import numpy as np


if __name__ == '__main__':


    image_path = "Homework\docker_compose\image.png"

    with open(image_path, "rb") as img_file:
        files = {"file": img_file}
        data = {'model_name' : 'torch'}
        response = requests.post("http://localhost:5001/preprocess", files=files, data=data)

    print(response.json())


