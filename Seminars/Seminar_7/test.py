import requests
import numpy as np

url_og = "http://localhost:8000/v2/models/my_model/infer"
url_opt = "http://localhost:8000/v2/models/my_model_opt/infer"
url_torch = "http://localhost:8000/v2/models/my_model_torch/infer"

inputs = np.random.rand(1, 1, 28, 28).astype(np.float32)
data = {
    "inputs": [
        {
            "name": "input",
            "shape": [1, 1, 28, 28],
            "datatype": "FP32",
            "data": inputs.flatten().tolist()
        }
    ]
}

response = requests.post(url_og, json=data)
print('original onnx model:')
print(response.status_code)
outputs = response.json()
# print(outputs)
print(outputs['outputs'][0]['data'])

response = requests.post(url_opt, json=data)
print('\n\noptimized onnx model:')
print(response.status_code)
outputs = response.json()
print(outputs['outputs'][0]['data'])

response = requests.post(url_torch, json=data)
print('\n\ntorch model:')
print(response.status_code)
outputs = response.json()
print(outputs['outputs'][0]['data'])
