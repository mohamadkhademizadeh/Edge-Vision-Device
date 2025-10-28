import cv2, numpy as np

def draw_boxes(image_bgr, boxes, labels=None):
    out = image_bgr.copy()
    if labels is None:
        labels = ['' for _ in boxes]
    for (x1,y1,x2,y2), lab in zip(boxes, labels):
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        if lab:
            ((tw, th), _) = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1-th-6), (x1+tw+4, y1), (0,255,0), -1)
            cv2.putText(out, lab, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return out
