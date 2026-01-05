import cv2
import numpy as np
from rembg import bg

# -----------------------------
# 1. Læs billede og fjern baggrund
# -----------------------------
img = cv2.imread("images/Daniel_billeder/IMG_0331.jpeg")
img = cv2.resize(img, (400, 600))

# Fjern baggrund med rembg
result_rgba = bg.remove(img)  # output BGRA

# Lav mask fra alfa-kanal
alpha = result_rgba[:, :, 3] / 255.0
mask = (alpha > 0.5).astype(np.uint8)

# -----------------------------
# 2. Find ansigt med Haar Cascade
# -----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Konverter til gråt billede til detektion
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

if len(faces) == 0:
    print("Ingen ansigt fundet!")
else:
    # Tag første ansigt (hvis der er flere)
    x, y, w, h = faces[0]

    # -----------------------------
    # 3. Brug mask til kun at tage forgrund i ansigts-bbox
    # -----------------------------
    face_mask = mask[y:y+h, x:x+w]
    face_crop = img[y:y+h, x:x+w]
    
    # Maskér baggrund inden for ansigtsboks
    face_only = cv2.bitwise_and(face_crop, face_crop, mask=face_mask.astype(np.uint8))

    # -----------------------------
    # 4. Gem ansigtet som fil
    # -----------------------------
    cv2.imwrite("face_only.png", face_only)
    print("Ansigt gemt som 'face_only.png'")

    # Valgfrit: vis
    cv2.imshow("Face Only", face_only)
    cv2.waitKey(0)
    cv2.destroyAllWindows()