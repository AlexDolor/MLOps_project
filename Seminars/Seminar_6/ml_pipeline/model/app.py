from fastapi import FastAPI, Request
import onnxruntime as ort, numpy as np
import requests

app = FastAPI()
session = ort.InferenceSession("model.onnx")

@app.post("/predict")

async def predict(req: Request):
    data = await req.json()
    x = np.array(data["tensor"], dtype=np.float32)#[None, ...]
    output = session.run(None, {"input": x})[0]
    payload = {"result": output.tolist()}
    r = requests.post("http://postprocess:5000/postprocess", json=payload)
    return r.json()