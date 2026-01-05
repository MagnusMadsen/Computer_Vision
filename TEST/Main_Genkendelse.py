from __future__ import print_function
import cv2
import numpy as np
import os
import argparse

# -----------------------------
# Paths
# -----------------------------
IMAGE_DIR = "../images/Daniel_billeder"
CASCADE_PATH = cv.data.haarcascades + "haarcascade_frontalface_default.xml"

# -----------------------------
# Argument parsing (kamera)
# -----------------------------
parser = argparse.ArgumentParser(description="Face recognition with LBPH")
parser.add_argument("--camera", help="Camera device number", default=0, type=int)
args = parser.parse_args()

# -----------------------------
# Initialize face detector
# -----------------------------
face_cascade = cv.CascadeClassifier(CASCADE_PATH)

# -----------------------------
# Load training images
# -----------------------------
faces = []
labels = []

for filename in os.listdir(IMAGE_DIR):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg", ".webp")):
        continue

    path = os.path.join(IMAGE_DIR, filename)
    img = cv.imread(path)

    if img is None:
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    detected = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in detected:
        face = gray[y:y+h, x:x+w]
        face = cv.resize(face, (200, 200))
        faces.append(face)
        labels.append(0)  # Daniel

if len(faces) == 0:
    raise RuntimeError("Ingen ansigter fundet i træningsbillederne")

# -----------------------------
# Train recognizer
# -----------------------------
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

print(f"Træning færdig – {len(faces)} ansigter indlæst")

# -----------------------------
# Video capture
# -----------------------------
cap = cv.VideoCapture(args.camera)
window_name = "Face Recognition"
cv.namedWindow(window_name)

while True:
    ret, frame = cap.read()
    if frame is None:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detected = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in detected:
        face = gray[y:y+h, x:x+w]
        face = cv.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        if confidence < 60:
            text = f"Daniel ({confidence:.1f})"
            color = (0, 255, 0)
        else:
            text = f"Ukendt ({confidence:.1f})"
            color = (0, 0, 255)

        cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv.putText(
            frame,
            text,
            (x, y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv.LINE_AA
        )

    cv.imshow(window_name, frame)

    key = cv.waitKey(30)
    if key == ord("q") or key == 27:
        break

cap.release()
cv.destroyAllWindows()
