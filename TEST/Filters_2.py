import os
import cv2
import random
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import numpy as np
from rembg import remove

# OpenCV DNN face detector (more robust than Haar)
DNN_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
DNN_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# Input videos
video_paths = [
    "Ansigt_utræk/Videoer/Daniel.mp4",
    "Ansigt_utræk/Videoer/Magnus.mp4",
]

# How many screenshots per video
num_screenshots = 10

output_size = (512, 512)
# Always save screenshots next to this script (independent of where you run it from)
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "screenshots")
os.makedirs(output_dir, exist_ok=True)

# Make video_path robust regardless of working directory
if not os.path.isabs(video_paths[0]):
    video_paths = [os.path.join(script_dir, vp) for vp in video_paths]

models_dir = os.path.join(script_dir, "models")
os.makedirs(models_dir, exist_ok=True)
proto_path = os.path.join(models_dir, "deploy.prototxt")
model_path = os.path.join(models_dir, "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# Download models if missing
if not os.path.exists(proto_path):
    print(f"[INFO] Downloading DNN prototxt to {proto_path}")
    urllib.request.urlretrieve(DNN_PROTO_URL, proto_path)
if not os.path.exists(model_path):
    print(f"[INFO] Downloading DNN caffemodel to {model_path}")
    urllib.request.urlretrieve(DNN_MODEL_URL, model_path)

try:
    dnn_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load DNN face detector: {e}")

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(cascade_path)
if haar_cascade.empty():
    raise RuntimeError("Cannot load Haar cascade")

saved_count = 0

# Count how many files we actually save
from threading import Lock
saved_lock = Lock()

def inc_saved():
    global saved_count
    with saved_lock:
        saved_count += 1

def detect_faces_dnn(frame, conf_threshold=0.55):
    """Return list of (x, y, w, h, conf) boxes using OpenCV DNN SSD face detector."""
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    dnn_net.setInput(blob)
    detections = dnn_net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < conf_threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        if bw > 0 and bh > 0:
            boxes.append((x1, y1, bw, bh, conf))

    # Sort by confidence then area
    boxes.sort(key=lambda b: (b[4], b[2] * b[3]), reverse=True)
    return boxes


def detect_faces_haar(frame):
    """Return list of (x, y, w, h) from Haar cascade."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = haar_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return faces

def process_frame(frame, file_idx):
    # 1) Try DNN detector first
    dnn_boxes = detect_faces_dnn(frame, conf_threshold=0.50)

    if dnn_boxes:
        x, y, w, h, conf = dnn_boxes[0]
        src = "DNN"
    else:
        # 2) Fallback to Haar
        faces = detect_faces_haar(frame)
        if len(faces) == 0:
            print(f"[WARN] No face detected for file_idx={file_idx}")
            return
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        conf = None
        src = "Haar"

    # Add padding around the face (more forgiving)
    pad_w = int(0.25 * w)
    pad_h = int(0.35 * h)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(frame.shape[1], x + w + pad_w)
    y2 = min(frame.shape[0], y + h + pad_h)

    face_crop = frame[y1:y2, x1:x2]

    # Remove background (rembg) and keep alpha channel
    try:
        face_rgba = remove(face_crop)
        if face_rgba is None or len(face_rgba.shape) != 3 or face_rgba.shape[2] != 4:
            raise ValueError("rembg did not return RGBA")
        alpha = face_rgba[:, :, 3]
        b, g, r = cv2.split(face_crop)
        face_png = cv2.merge([b, g, r, alpha])
    except Exception as e:
        print(f"[WARN] rembg failed for file_idx={file_idx} ({src}): {e} — saving without alpha")
        face_png = face_crop

    face_png = cv2.resize(face_png, output_size, interpolation=cv2.INTER_LINEAR)

    save_path = os.path.join(output_dir, f"face_{start_index + file_idx}.png")
    ok = cv2.imwrite(save_path, face_png)
    if not ok:
        print(f"[ERROR] Failed to write: {save_path}")
    else:
        global saved_count
        saved_count += 1
        print(f"[OK] Saved: {save_path}")

start_index = 0

for video_path in video_paths:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)

    # On macOS, depending on how OpenCV was built, some backends may work better
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(video_path, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print(f"OpenCV version: {cv2.__version__}")
        print(f"Tried to open: {video_path}")
        print("If this MP4 still won't open, re-encode it to H.264/AAC (see comment below).")
        raise RuntimeError("Cannot open video! (OpenCV could not read the stream)")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video has 0/unknown frames (cannot decode). Try re-encoding to H.264/AAC.")

    # Choose random frames in this video
    frame_indices = sorted(random.sample(range(total_frames), min(num_screenshots, total_frames)))
    frame_idx_set = set(frame_indices)
    frame_pos_map = {fi: i for i, fi in enumerate(frame_indices)}

    # Tag filenames by video name
    video_tag = os.path.splitext(os.path.basename(video_path))[0]

    frames_to_process = []
    frame_idx = 0
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 🔥 ROTER 90° TIL HØJRE (VIGTIG)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # If the frame is small, upscale a bit to help detection
        h, w = frame.shape[:2]
        if max(h, w) < 720:
            scale = 720 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        if frame_idx in frame_idx_set:
            file_idx = frame_pos_map[frame_idx]
            frames_to_process.append((frame.copy(), file_idx))

        frame_idx += 1

    cap.release()

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(lambda args: process_frame(*args), frames_to_process)

    start_index += len(frames_to_process)

print(f"Done! Requested={num_screenshots}, queued={len(frames_to_process)}, saved={saved_count}. Output: {output_dir}")
