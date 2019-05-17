
import cv2
import face_detect
import face_landmarks
import time

#Start the camera twice to work around
# "error: ..\..\..\modules\imgproc\src\color.cpp:7456: error: (-215)
# scn == 3 || scn == 4 in function cv::ipp_cvtColor"

## my camera -- HxW == 480 x 640
face_selected = False


def handle_event(event, x, y, flags, param):
    global face_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        in_var = input("Want to learn this face?")

        if (in_var == 'y' or in_var == 'yes'):
            bboxes = detector.detect_faces(frame, frame.shape, 0.8)
            if bboxes:
                aligned_face = marker.align_marked_faces(frame_orig, bboxes[0])
                in_name = input('What is the person\'s name?')
                cv2.imwrite('./resources/faces/' + str(in_name) + '.png', aligned_face)
            else:
                print('No face detected!')

        elif (in_var == 'q'):
            print('Cancel\n')
            camera_capture.release()
            cv2.destroyAllWindows()


cv2.namedWindow('FaceApp')
cv2.setMouseCallback('FaceApp', handle_event)

camera_capture = cv2.VideoCapture(0)
camera_capture.release()
camera_capture = cv2.VideoCapture(0)

# if detector == DNN :
# else if detector == HAAR :
detector = face_detect.FaceDetectionDNN()
marker = face_landmarks.Marker()

while True:
    if not camera_capture.isOpened():
        print('Unable to load camera.')
        break


    # Capture frame-by-frame
    ret, frame = camera_capture.read()

    bboxes = detector.detect_faces(frame, frame.shape, 0.8)
    frame_orig = frame.copy()

    for bb in bboxes:
        (x, y, w, h) = bb

        cv2.rectangle(frame, (x, y, w, h), (255, 255, 0), 2)

    cv2.imshow('FaceApp', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything is done, release the capture
camera_capture.release()
cv2.destroyAllWindows()
