
import cv2
import face_detect
import face_landmarks
import face_recognition

#Start the camera twice to work around
# "error: ..\..\..\modules\imgproc\src\color.cpp:7456: error: (-215)
# scn == 3 || scn == 4 in function cv::ipp_cvtColor"

## my camera -- HxW == 480 x 640


camera_capture = cv2.VideoCapture(0)
camera_capture.release()
camera_capture = cv2.VideoCapture(0)

# if detector == DNN :
# else if detector == HAAR :
detector = face_detect.FaceDetectionDNN()
marker = face_landmarks.Marker()
recognizer = face_recognition.FaceRecognitionDNN()


while True:
    if not camera_capture.isOpened():
        print('Unable to load camera.')
        break

    # Capture frame-by-frame
    ret, frame = camera_capture.read()


    # loading saved embeddings of known faces

    bboxes = detector.detect_faces(frame, frame.shape, 0.8)

    for (x,y,w,h) in bboxes:
        cv2.rectangle(frame, (x, y, w, h ), (255,255,0),2)

    aligned_faces = []
    for bb in bboxes:
        aligned_face = marker.align_marked_faces(frame, bb)
        aligned_faces.append(aligned_face)

    counter = 0
    for face in aligned_faces:
        name = recognizer.recognize(face)

        if(name != None):
            (x,y,w,h) = bboxes[counter]
            mg = cv2.rectangle(frame, (x,y,w,h), (255, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(name), (x + 5, y - 5), font, 1, (255, 255, 0), 2)

        counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
camera_capture.release()
cv2.destroyAllWindows()
