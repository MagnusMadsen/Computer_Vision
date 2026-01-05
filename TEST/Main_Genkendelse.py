import cv2
import numpy as np
import os

# -----------------------------
# Paths
# -----------------------------
IMAGE_DIR = "../images/Daniel_billeder"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# -----------------------------
# Initialize
# -----------------------------
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []

# -----------------------------
# Load and process images
# -----------------------------
for filename in os.listdir(IMAGE_DIR):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg", ".webp")):
        continue

    img_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(img_path)

    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in detected_faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        faces.append(face)
        labels.append(0)  # Samme person (Daniel)

        # Visual debug (kan kommenteres ud)
        cv2.imshow("Detected face", face)
        cv2.waitKey(300)

cv2.destroyAllWindows()

# -----------------------------
# Train model
# -----------------------------
if len(faces) == 0:
    raise RuntimeError("Ingen ansigter fundet i billedmappen")

recognizer.train(faces, np.array(labels))
print(f"Træning gennemført med {len(faces)} ansigter")

# -----------------------------
# Test recognition
# -----------------------------
for i, face in enumerate(faces):
    label, confidence = recognizer.predict(face)
    print(f"Billede {i}: Label={label}, Confidence={confidence:.2f}")

    if confidence < 60:
        print("→ Samme person (match)")
    else:
        print("→ Usikker / ukendt")
