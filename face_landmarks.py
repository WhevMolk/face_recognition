import sys
import dlib
import numpy as np
import cv2
import openface
from skimage import io



class Marker(object):

    def __init__(self):
        self.predictor_model = "./resources/face_landmarks/shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(self.predictor_model)
        self.aligner = openface.AlignDlib(self.predictor_model)

    # transforms the dlib rectangle format to the opencv representation
    @staticmethod
    def transform_rect_to_bb(rect):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y

        # return a tuple of (x, y, w, h)
        return (x, y, w, h)

    def transform_bb_to_rect(self, bb):
            return dlib.rectangle(left=bb[0], top=bb[1], right=(bb[2] + bb[0]), bottom=(bb[3] + bb[1]))

    def shape_to_np(shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)

        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords

    def _get_landmarks(self, img, bboxes):
        all_face_landmarks = []

        for bb in bboxes:
                face_landmark = self.aligner.findLandmarks(img, self.transform_bb_to_rect(bb))
                all_face_landmarks.append(face_landmark)
        return all_face_landmarks


    def show_marked_faces(self, img, bboxes):
        all_landmarks = self._get_landmarks(img, bboxes)

        landmarks_img = np.zeros(img.shape, dtype=np.uint8)
        for lm in all_landmarks:
            for (x, y) in lm:
                cv2.circle(landmarks_img, (x, y), 1, (0, 0, 255), -1)
        return landmarks_img


    def align_marked_faces(self, img, bb):

        aligned = self.aligner.align(imgDim=96, rgbImg= img, bb=self.transform_bb_to_rect(bb), landmarkIndices= \
                                                      openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
        return aligned