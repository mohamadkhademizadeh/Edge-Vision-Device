# Edge-Vision-Device

**Smart edge vision** device that runs on **Raspberry Pi (ARM)** or **NVIDIA Jetson**.
- Lightweight **object/defect detection** (YOLOv8n by default; optional TensorRT/TFLite)
- **Camera capture** from CSI/USB/webcam with FPS throttling
- **MQTT publisher** (pass/fail & metrics) and optional **Modbus/TCP client**
- **FastAPI service** for remote inference & health checks
- **Streamlit dashboard** to view the live feed and results locally
- Dockerfiles for **x86 dev**, **Raspberry Pi**, and **Jetson**

> Goal: demonstrate your ability to bridge **AI + embedded + industrial integration**.

---

## Quickstart (dev laptop)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# run the local dashboard (uses your webcam if the CSI/USB cam is missing)
streamlit run app/dashboard.py
```

### On Raspberry Pi / Jetson
- Install Python 3.10+ and dependencies (on Jetson you can also use the Dockerfile).
- Configure the camera (e.g., `/dev/video0`) in `configs/device.yaml`.
- Start the service:
```bash
python -m edgevision.run --config configs/device.yaml
```
- Or use **systemd** (see `deploy/systemd/edgevision.service`).

---

## Features
- Pluggable **backends**: ultralytics YOLO, or a built-in **classical fallback** (edges/contours).
- **I/O adapters**:
  - MQTT publisher (topic: `edgevision/events` + retained status topic).
  - Modbus/TCP client (coil write example for pass/fail bit).
  - REST API (FastAPI) for `/infer`, `/health`, `/config`.
- **Metrics**: simple FPS / latency logs and per-frame JSON output.
- **Safe defaults**: runs even without GPU/model by using fallback path.

---

## Repo Layout
```
Edge-Vision-Device/
├── app/
│   └── dashboard.py           # Streamlit local UI
├── configs/
│   └── device.yaml            # Camera, backend, MQTT, Modbus, API
├── deploy/
│   ├── docker/
│   │   ├── Dockerfile.jetson
│   │   └── Dockerfile.dev
│   └── systemd/
│       └── edgevision.service
├── models/                    # place weights here (e.g., yolov8n.pt)
├── scripts/
│   ├── export_tensorrt.md     # notes for TRT export on Jetson
│   └── export_tflite.md       # notes for TFLite export on Pi
├── src/edgevision/
│   ├── __init__.py
│   ├── run.py                 # main loop (camera -> detect -> publish -> serve)
│   ├── service.py             # FastAPI
│   ├── backends/
│   │   ├── yolo.py
│   │   └── classical.py
│   └── io/
│       ├── camera.py
│       ├── mqtt_pub.py
│       ├── modbus_client.py
│       └── viz.py
├── tests/
│   └── test_classical.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## Roadmap
- [ ] TensorRT runtime class with engine caching
- [ ] TFLite Micro for microcontrollers (esp32-cam demo)
- [ ] Hardware GPIO outputs (Raspberry Pi) for tower-light / relay
- [ ] On-device recording of fails for re-training
