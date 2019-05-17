import cv2
import dlib
import numpy as np

class FaceDetectionHAAR(object):
    def __init__(self):
        self.casc_path = "./resources/face_detect/haarcascade_frontalface_default.xml"
        self.casc = cv2.CascadeClassifier(self.casc_path)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.casc.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        return frame

class FaceDetectionDNN(object):
    def __init__(self):
        modelFile = "./resources/face_detect/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "./resources/face_detect/deploy.prototxt"
        self.net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    def detect_faces(self, frame, shape, thresh):
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)

        self.net.setInput(blob)
        detections = self.net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > thresh:
                x1 = int(detections[0, 0, i, 3] * shape[1])
                y1 = int(detections[0, 0, i, 4] * shape[0])
                x2 = int(detections[0, 0, i, 5] * shape[1])
                y2 = int(detections[0, 0, i, 6] * shape[0])
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0),2)
                bboxes.append([x1, y1, x2-x1, y2-y1])
        return bboxes

    def cut_face(self, frame, bb):
        img = np.zeros(frame.shape, dtype=np.uint8)
        (x,y,w,h) = bb
        return frame[y:y+h, x:x+w]


class FaceDetectionDLIB(object):

    def detect_faces(self, frame):
       pass
