import cv2

casc_path_face = "./resources/face_detect/haarcascade_frontalface_default.xml"

casc_face = cv2.CascadeClassifier(casc_path_face)

# Start the camera twice to work around
# "error: ..\..\..\modules\imgproc\src\color.cpp:7456: error: (-215)
# scn == 3 || scn == 4 in function cv::ipp_cvtColor"
camera_capture = cv2.VideoCapture(0)
camera_capture.release()
camera_capture = cv2.VideoCapture(0)

detect_eyes = True
detect_smile = True

while True:
    if not camera_capture.isOpened():
        print('Unable to load camera.')
        break

    # Capture frame-by-frame
    ret, frame = camera_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = casc_face.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(30, 30)
    )
    # Draw a rectangle around the faces

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
camera_capture.release()
cv2.destroyAllWindows()