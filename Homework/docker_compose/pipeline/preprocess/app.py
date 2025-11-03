from fastapi import FastAPI, UploadFile, Form
import requests, numpy as np
from PIL import Image

app = FastAPI()
@app.post("/preprocess")
async def process_image(file: UploadFile, model_name: str = Form(...)):
    # payload = {"tensor": arr.tolist()}

    triton_url = 'http://triton:8000/v2/models/'
    model_url = ''
    match model_name:
        case 'torch':
            model_url = triton_url + 'default_model' +'/infer'
        case 'torch_quant':
            model_url = triton_url + 'quant_model' +'/infer'
        case 'onnx':
            model_url = triton_url + 'onnx_model' +'/infer'
        case 'onnx_opt':
            model_url = triton_url + 'onnx_opt_model' +'/infer'
        case _:
            return {'error':f'No such model exists ({model_name})'}
    
    image = Image.open(file.file)
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = arr.reshape((1,1,28,28))
    data = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, 1, 28, 28],
                "datatype": "FP32",
                "data": arr.flatten().tolist()
            }
        ]
    }
    
    r = requests.post(model_url, json=data)
    outputs = r.json()

    data = {'result': [outputs['outputs'][0]['data']]}
    r = requests.post("http://postprocess:5000/postprocess", json=data)
    # print(outputs['outputs'][0]['data'])
    # return outputs['outputs'][0]['data']
    return r.json()