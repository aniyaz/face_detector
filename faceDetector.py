import cv2
from random import randrange

cv2.namedWindow("Face Detector", cv2.WINDOW_NORMAL)
#Load data to train
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Read Image
#img = cv2.imread('fam1.jpg')

#Capture webcam
webcam = cv2.VideoCapture(0)

#Iterate forever over frames
while True:

    #read the current frame
    successful_frame_read, frame = webcam.read()

    #Cover to grayscale Image
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(frame)

    #Draw rectangle around the face
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(125, 256),randrange(125, 256),randrange(125, 256)), 4)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    #Stop if Q/q is pressed.
    if key==81 or key==113:
        break

webcam.release()
