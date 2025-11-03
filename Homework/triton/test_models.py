import requests
import numpy as np


def check_model(url, data):
    response = requests.post(url, json=data)
    print(response.status_code)
    outputs = response.json()
    print(outputs)
    print(outputs['outputs'][0]['data'])
    return

if __name__ == '__main__':
    url_torch = "http://localhost:8000/v2/models/default_model/infer"
    url_onnx = "http://localhost:8000/v2/models/onnx_model/infer"
    url_onnx_opt = "http://localhost:8000/v2/models/onnx_opt_model/infer"
    url_torch_quant = "http://localhost:8000/v2/models/quant_model/infer"


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

    print('\noriginal torch model:')
    check_model(url_torch, data)
    
    print('\nquant torch model:')
    check_model(url_torch_quant, data)

    print('\nonnx model:')
    check_model(url_onnx, data)

    print('\notimized onnx model:')
    check_model(url_onnx_opt, data)
