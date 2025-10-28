import numpy as np, cv2
from edgevision.backends.classical import ClassicalBackend

def test_rects():
    img = np.zeros((200,200,3), np.uint8)
    cv2.rectangle(img, (20,20), (100,120), (255,255,255), -1)
    b,l,s = ClassicalBackend(min_area=50).infer(img)
    assert len(b) >= 1
