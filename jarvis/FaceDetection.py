import cv2
import os

resourcesPath = os.path.dirname(__file__) + '/resources/'


class FaceDetection:

    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.cam.set(cv2.CAP_PROP_FPS, 1)

    def run(self):
        while True:
            ret, frame = self.cam.read()

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_frontal_cascade = cv2.CascadeClassifier(
                resourcesPath + 'haarcascades/haarcascade_frontalface_default.xml')
            faces = face_frontal_cascade.detectMultiScale(frame_gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow('face detection', frame)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

    def __exit__(self, exc_type, exc_val, exc_tb):
        cv2.destroyAllWindows()
