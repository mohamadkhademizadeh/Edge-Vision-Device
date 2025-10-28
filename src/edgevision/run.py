import os, time, json, threading, yaml, cv2
from .io.camera import Camera
from .io.viz import draw_boxes
from .io.mqtt_pub import MQTTPublisher
from .io.modbus_client import ModbusClient
from .backends.yolo import YOLOBackend
from .backends.classical import ClassicalBackend
from .service import run_api

def load_config(path):
    with open(path,'r') as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config(os.environ.get('EDGEVISION_CONFIG','configs/device.yaml'))
    cam = Camera(cfg['camera']['index'], cfg['camera']['width'], cfg['camera']['height'], cfg['camera']['fps'])

    # backend selection
    backend = cfg['inference']['backend']
    engine = None
    if backend == 'yolo':
        try:
            engine = YOLOBackend(cfg['inference']['model_path'], cfg['inference']['conf_thres'], cfg['inference']['iou_thres'])
        except Exception as e:
            print("[WARN] YOLO unavailable:", e, "-> using classical fallback")
            engine = ClassicalBackend()
    elif backend == 'classical':
        engine = ClassicalBackend()
    else:  # auto
        try:
            engine = YOLOBackend(cfg['inference']['model_path'], cfg['inference']['conf_thres'], cfg['inference']['iou_thres'])
        except Exception:
            engine = ClassicalBackend()

    # MQTT
    mqtt = None
    if cfg['mqtt']['enabled']:
        mqtt = MQTTPublisher(cfg['mqtt']['host'], cfg['mqtt']['port'], cfg['mqtt']['topic'], cfg['mqtt']['status_topic'])

    # Modbus
    mb = None
    if cfg['modbus']['enabled']:
        mb = ModbusClient(cfg['modbus']['host'], cfg['modbus']['port'], cfg['modbus']['coil_address'])

    # API thread
    api_thread = threading.Thread(target=run_api, kwargs={'host':cfg['api']['host'], 'port':cfg['api']['port']}, daemon=True)
    api_thread.start()

    try:
        while True:
            frame = cam.read()
            if frame is None:
                print("[WARN] camera returned None frame")
                time.sleep(0.2)
                continue

            t0 = time.time()
            boxes, labels, scores = engine.infer(frame)
            latency = (time.time()-t0)*1000.0
            ok = len(boxes) == 0  # simple pass/fail: no detections -> pass
            if mb:
                try: mb.write_pass_fail(ok)
                except Exception as e: print("[WARN] Modbus write failed:", e)
            if mqtt:
                payload = {'ok': ok, 'num_boxes': len(boxes), 'latency_ms': latency, 'ts': time.time()}
                mqtt.publish_event(payload)

            vis = draw_boxes(frame, boxes, labels)
            cv2.imshow("EdgeVision", vis)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        if mqtt: mqtt.close()
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
