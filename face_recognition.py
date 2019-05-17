import numpy as np
import glob
import os
import cv2

from model import create_model

class FaceRecognitionDNN(object):
    def __init__(self):
        weightsFile = "./resources/face_recognition/model/nn4.small2.v1.h5"
        self.net = create_model()
        self.net.load_weights('./resources/face_recognition/model/nn4.small2.v1.h5')

    def predict(self, img):
        return self.net.predict(np.expand_dims(img, axis=0))[0]


    def recognize(self,img):

        minimum_distance = 100
        person_name = None
        emb1 = self.predict(img)

        for file in glob.glob('./resources/faces/*'):
            image_file = cv2.imread(file, 1)

            emb2 = self.predict(image_file)
            distance = np.linalg.norm(emb1 - emb2)
            print('Distance to {}: {}\n'.format(os.path.splitext(os.path.basename(file))[0], distance))
            if (distance < minimum_distance):
                person_name = os.path.splitext(os.path.basename(file))[0]
                minimum_distance = distance



        if distance < 0.5:
            print('successfully recognized: {}!'.format(person_name))
            return person_name
        else:
            return None