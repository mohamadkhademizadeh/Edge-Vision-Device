import cv2, time

class Camera:
    def __init__(self, index=0, width=640, height=480, fps=10):
        self.index = index
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.period = 1.0 / max(1, fps)
        self.last_t = 0.0

    def read(self):
        # fps throttle
        now = time.time()
        wait = self.period - (now - self.last_t)
        if wait > 0:
            time.sleep(wait)
        self.last_t = time.time()
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass
