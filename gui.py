
import cv2
import face_detect
import face_landmarks
from tkinter import *

#Start the camera twice to work around
# "error: ..\..\..\modules\imgproc\src\color.cpp:7456: error: (-215)
# scn == 3 || scn == 4 in function cv::ipp_cvtColor"

## my camera -- HxW == 480 x 640

class popupWindow(object):
    def __init__(self,master):
        top = self.top=Toplevel(master)
        self.e = Entry(top)
        self.e.pack()
        self.b = Button(top,text='Ok',command=self.cleanup)
        self.b.pack()
    def cleanup(self):
        self.value=self.e.get()
        self.top.destroy()

class mainWindow(object):
    def __init__(self, master):
        self.master = master
        self.l = Label(text="FaceApp")
        self.b = Button(master, text="Learn face", command=self.popup_learn)
        self.b.pack()
        self.b2 = Button(master, text="Recognize faces", command=self.popup_recognize)
        self.b2.pack()

    def popup_learn(self):
        learn_face()

    def popup_recognize(self):
        recognize_faces()

    def entryValue(self):
        return self.w.value


def init_setup():
    pass

def learn_face():
    pass

def recognize_faces():
    pass


if __name__ == "__main__":
    root = Tk()
    m = mainWindow(root)
    root.mainloop()

