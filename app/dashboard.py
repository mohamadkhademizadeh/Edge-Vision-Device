import streamlit as st
import cv2, numpy as np, time, yaml, json, os
from PIL import Image
from edgevision.io.camera import Camera
from edgevision.io.viz import draw_boxes
from edgevision.backends.yolo import YOLOBackend
from edgevision.backends.classical import ClassicalBackend

st.set_page_config(page_title="Edge Vision Device", layout="wide")
st.title("📟 Edge Vision Device — Local Dashboard")

with open('configs/device.yaml','r') as f:
    CFG = yaml.safe_load(f)

backend = st.sidebar.selectbox("Backend", ["auto","yolo","classical"], index=0)
CFG['inference']['backend'] = backend

if backend == "yolo":
    model_path = st.sidebar.text_input("Model path", CFG['inference']['model_path'])
    CFG['inference']['model_path'] = model_path

st.sidebar.write("Camera")
cam_index = st.sidebar.number_input("Camera index", 0, 10, CFG['camera']['index'])
width = st.sidebar.number_input("Width", 160, 1920, CFG['camera']['width'], step=10)
height = st.sidebar.number_input("Height", 120, 1080, CFG['camera']['height'], step=10)
fps = st.sidebar.number_input("FPS", 1, 60, CFG['camera']['fps'])

start = st.button("Start")

if start:
    cam = Camera(cam_index, width, height, fps)
    if backend == "yolo":
        try:
            engine = YOLOBackend(CFG['inference']['model_path'], CFG['inference']['conf_thres'], CFG['inference']['iou_thres'])
        except Exception as e:
            st.warning(f"YOLO backend not available ({e}); switching to classical.")
            engine = ClassicalBackend()
    elif backend == "classical":
        engine = ClassicalBackend()
    else:
        try:
            engine = YOLOBackend(CFG['inference']['model_path'], CFG['inference']['conf_thres'], CFG['inference']['iou_thres'])
        except Exception:
            engine = ClassicalBackend()

    ph = st.empty()
    t0 = time.time(); frames = 0
    try:
        while True:
            frame = cam.read()
            if frame is None:
                st.warning("Camera frame is None; check camera index.")
                time.sleep(0.2)
                continue
            boxes, labels, scores = engine.infer(frame)
            vis = draw_boxes(frame[..., ::-1], boxes, labels)  # BGR->RGB for display
            frames += 1
            if frames % 10 == 0:
                fps_est = frames / (time.time() - t0 + 1e-6)
                st.sidebar.metric("FPS", f"{fps_est:.1f}")
                st.sidebar.metric("Detections", f"{len(boxes)}")
            ph.image(vis, use_column_width=True)
    except KeyboardInterrupt:
        pass
