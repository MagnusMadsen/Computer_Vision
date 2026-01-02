from __future__ import print_function
import cv2 as cv
import argparse

face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

window_capture_name = 'Video Capture'

parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()

cap = cv.VideoCapture(args.camera)

cv.namedWindow(window_capture_name)

while True:

    ret, frame = cap.read()
    if frame is None:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv.imshow(window_capture_name, frame)

    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break