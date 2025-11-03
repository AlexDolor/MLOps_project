import requests, numpy as np
from PIL import Image

image = Image.open('Seminars\Seminar_6\image.png')#.resize((28, 28))
arr = np.asarray(image).astype(np.float32) / 255.0
payload = {"tensor": arr.tolist()}
print(arr.reshape((1,28,28)).shape)
print(arr.tolist())