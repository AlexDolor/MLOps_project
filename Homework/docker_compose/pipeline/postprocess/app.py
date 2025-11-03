from fastapi import FastAPI, Request
import onnxruntime as ort, numpy as np

app = FastAPI()

@app.post("/postprocess")

async def predict(req: Request):
    data = await req.json()
    x = np.argmax(data['result'], 1).tolist()
    return {"result_class": x}