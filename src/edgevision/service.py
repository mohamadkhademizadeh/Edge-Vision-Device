import os, io, time, threading, json
from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2, numpy as np

from .backends.yolo import YOLOBackend
from .backends.classical import ClassicalBackend

app = FastAPI()
state = {
    'backend':'classical',
    'engine': ClassicalBackend(),
}

@app.get('/health')
def health():
    return {'status':'ok', 'backend': state['backend']}

@app.post('/config')
def set_config(backend: str = 'classical', model_path: str = 'models/yolov8n.pt'):
    if backend == 'yolo':
        engine = YOLOBackend(model_path)
    else:
        engine = ClassicalBackend()
    state['backend'] = backend
    state['engine'] = engine
    return {'status':'ok', 'backend': backend}

@app.post('/infer')
async def infer(file: UploadFile = File(...)):
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    boxes, labels, scores = state['engine'].infer(img)
    return {'boxes': boxes, 'labels': labels, 'scores': scores}

def run_api(host='0.0.0.0', port=8000):
    uvicorn.run(app, host=host, port=port)
