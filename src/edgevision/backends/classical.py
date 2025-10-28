import cv2, numpy as np

class ClassicalBackend:
    def __init__(self, min_area=300, c1=50, c2=150):
        self.min_area = min_area; self.c1 = c1; self.c2 = c2

    def infer(self, image_bgr):
        g = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        e = cv2.Canny(g, self.c1, self.c2)
        e = cv2.dilate(e, np.ones((3,3), np.uint8), iterations=1)
        contours,_ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes=[]; labels=[]; scores=[]
        for c in contours:
            a = cv2.contourArea(c)
            if a < self.min_area:
                continue
            x,y,w,h = cv2.boundingRect(c)
            boxes.append((x,y,x+w,y+h))
            labels.append("candidate")
            scores.append(0.5)
        return boxes, labels, scores
