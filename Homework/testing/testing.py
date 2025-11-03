import requests
# import torchvision
import torchvision.transforms as transforms
# import torch
# import numpy as np
import os
from PIL import Image
import time
import pandas as pd
from tqdm.auto import tqdm
import subprocess

def check_model(image_paths, model_name, n=10):
    key = "result_class"
    start = time.perf_counter()
    for i, img_path in enumerate(tqdm(image_paths)):
        with open(img_path, "rb") as img_file:
            files = {"file": img_file}
            data = {'model_name' : model_name}
            response = requests.post("http://localhost:5001/preprocess", files=files, data=data)
        resp_dict = response.json()
        
        if not(
            key in resp_dict 
            and isinstance(resp_dict[key], list) 
            and len(resp_dict[key]) == 1
            and isinstance(resp_dict[key][0], int)
        ): return 0, 0, {'status':'error', 'resp_dict':resp_dict}   
        if i >= n: break
    elapsed = time.perf_counter() - start
    obj_number = min(n, len(image_path))
    latency = (elapsed / obj_number) * 1000  # ms per request
    throughput = obj_number / elapsed
    return latency, throughput, {'status':'ok'}

if __name__ == '__main__':


    image_path = "Homework\\testing\\images\\"

    # image_list = load_images(image_path)
    
    image_paths = []
    for filename in os.listdir(image_path):
        file_path = os.path.join(image_path, filename)
        image_paths.append(file_path)

    models = ['torch','torch_quant','onnx', 'onnx_opt']
    metrics_df = pd.DataFrame(columns=['latency (ms)', 'throughput (obj/sec)'])
    n = 1000

    for model in models:
        result = check_model(image_paths, 'torch', n)
        if result[-1]['status'] == 'ok':
            metrics_df.loc[model] = result[:-1]
        else:
            print(result[-1]['resp_dict'])
            break
    
    containers = ['pipeline-preprocess-1','pipeline-triton-1', 'pipeline-postprocess-1']
    for container in containers:
        logs = subprocess.check_output(["docker", "logs", container]).decode()
        with open('Homework\\testing\\logs\\' + container + '.log', 'w') as f:
            f.write(logs)
    
    print(metrics_df)
    metrics_df.to_csv('Homework\\testing\\metrics.csv')


