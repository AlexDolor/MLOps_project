from fastapi import FastAPI, UploadFile
import requests, numpy as np
from PIL import Image

app = FastAPI()
@app.post("/process")
async def process_image(file: UploadFile):
    image = Image.open(file.file)
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = arr.reshape((1,1,28,28))
    payload = {"tensor": arr.tolist()}
    r = requests.post("http://model:5000/predict", json=payload)
    return r.json()