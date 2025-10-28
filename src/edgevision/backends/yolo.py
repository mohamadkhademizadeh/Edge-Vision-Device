import os, numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

class YOLOBackend:
    def __init__(self, weights, conf=0.25, iou=0.45):
        if YOLO is None:
            raise RuntimeError("Ultralytics not installed")
        if not os.path.exists(weights):
            raise FileNotFoundError(weights)
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou

    def infer(self, image_bgr):
        res = self.model(image_bgr, conf=self.conf, iou=self.iou)[0]
        boxes = []
        labels = []
        scores = []
        if hasattr(res, 'boxes'):
            xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy().tolist()
            clss = res.boxes.cls.cpu().numpy().astype(int).tolist()
            names = res.names if hasattr(res, 'names') else {i: f"class_{i}" for i in set(clss)}
            for (x1,y1,x2,y2), s, c in zip(xyxy, confs, clss):
                boxes.append((int(x1),int(y1),int(x2),int(y2)))
                labels.append(str(names.get(c, f"class_{c}")))
                scores.append(float(s))
        return boxes, labels, scores
