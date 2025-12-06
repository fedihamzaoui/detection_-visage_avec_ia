
import cv2
import numpy as np
import time
from pathlib import Path
import os
import math
import sys
import threading
import queue
import importlib

# Optional dependencies
try:
    import importlib
    mp = importlib.import_module("mediapipe")
    MEDIAPIPE_AVAILABLE = True
except Exception:
    mp = None
    MEDIAPIPE_AVAILABLE = False
try:
    # dynamic import to avoid static unresolved-import flags
    face_recognition = importlib.import_module("face_recognition")
    FACE_RECOG_AVAILABLE = True
except Exception:
    # Minimal fallback stub to satisfy name usage in the code and keep runtime safe.
    # The real functionality will remain disabled via FACE_RECOG_AVAILABLE = False.
    class _FaceRecogStub:
        def load_image_file(self, path):
            raise RuntimeError("face_recognition is not installed")
        def face_encodings(self, img):
            # return no encodings so the app treats it as no known faces
            return []
        def face_distance(self, known_encodings, encoding):
            # return large distances so no match will be found
            return [1.0 for _ in known_encodings]
    face_recognition = _FaceRecogStub()
    FACE_RECOG_AVAILABLE = False
    FACE_RECOG_AVAILABLE = False

try:
    # import tensorflow dynamically to avoid static "module not found" errors in editors/linters
    tf = importlib.import_module("tensorflow")
    # Avoid direct 'from tensorflow.keras.models import load_model' which some linters/static analyzers
    # cannot resolve; use attribute access on the already-imported tensorflow module.
    load_model = getattr(tf.keras.models, "load_model")
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False
    load_model = None

try:
    from scipy.signal import detrend, find_peaks
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# Paths and optional models
BASE = Path.cwd()
EMOTION_MODEL_P = BASE / "emotion_model.h5"
AGE_MODEL_P = BASE / "age_model.h5"
GENDER_MODEL_P = BASE / "gender_model.h5"
KNOWN_FACES_DIR = BASE / "known_faces"

# Haar cascades (local, no download)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Load optional advanced models
advanced_emotion = None
advanced_age = None
advanced_gender = None
if TF_AVAILABLE:
    if EMOTION_MODEL_P.exists():
        try:
            advanced_emotion = load_model(str(EMOTION_MODEL_P))
            print("Loaded emotion_model.h5")
        except Exception as e:
            print("Failed loading emotion model:", e)
    if AGE_MODEL_P.exists():
        try:
            advanced_age = load_model(str(AGE_MODEL_P))
            print("Loaded age_model.h5")
        except Exception as e:
            print("Failed loading age model:", e)
    if GENDER_MODEL_P.exists():
        try:
            advanced_gender = load_model(str(GENDER_MODEL_P))
            print("Loaded gender_model.h5")
        except Exception as e:
            print("Failed loading gender model:", e)

# Face recognition DB (optional)
known_encodings = []
known_names = []
if FACE_RECOG_AVAILABLE and KNOWN_FACES_DIR.exists():
    for img_path in KNOWN_FACES_DIR.glob("*.*"):
        name = img_path.stem
        try:
            img = face_recognition.load_image_file(str(img_path))
            encs = face_recognition.face_encodings(img)
            if encs:
                known_encodings.append(encs[0])
                known_names.append(name)
                print(f"Loaded known face: {name}")
        except Exception as e:
            print("Face load err", img_path, e)

# MediaPipe for pose/hands/face mesh (optional)
if MEDIAPIPE_AVAILABLE:
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh

# Helpers
EMOTION_LABELS = ["angry","disgust","fear","happy","sad","surprise","neutral"]

def preprocess_gray_for_emotion(img_gray, size=(48,48)):
    return cv2.resize(img_gray, size).astype("float32")/255.0.reshape((1,size[0],size[1],1))

def preprocess_color_for_age(img, size=(64,64)):
    return cv2.resize(img, size).astype("float32")/255.0.reshape((1,size[0],size[1],3))

def predict_emotion_advanced(face_gray):
    if advanced_emotion is None:
        return None
    try:
        x = cv2.resize(face_gray, (48,48)).astype("float32")/255.0
        x = np.expand_dims(x, axis=(0,-1))
        p = advanced_emotion.predict(x)
        if p.ndim == 2:
            idx = int(np.argmax(p[0]))
            return EMOTION_LABELS[idx], float(np.max(p[0]))
        return None
    except Exception as e:
        print("Emotion predict err", e)
        return None

def predict_age_advanced(face_color):
    if advanced_age is None:
        return None
    try:
        x = preprocess_color_for_age(face_color)
        p = advanced_age.predict(x)
        if p.ndim == 2 and p.shape[1] == 1:
            return int(float(p[0,0]))
        if p.ndim == 2:
            ages = np.arange(p.shape[1])
            return int(np.dot(p[0], ages))
        return None
    except Exception as e:
        print("Age predict err", e)
        return None

def predict_gender_advanced(face_color):
    if advanced_gender is None:
        return None
    try:
        x = preprocess_color_for_age(face_color, size=(64,64))
        p = advanced_gender.predict(x)
        if p.ndim == 2 and p.shape[1] == 1:
            return ("male" if p[0,0] > 0.5 else "female"), float(p[0,0])
        if p.ndim == 2:
            idx = int(np.argmax(p[0]))
            return ("male" if idx==1 else "female"), float(np.max(p[0]))
        return None
    except Exception as e:
        print("Gender predict err", e)
        return None

# Heuristic emotion: smile -> happy, else neutral/sad using mouth darkness
def heuristic_emotion(face_gray, face_color):
    smiles = smile_cascade.detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=20, minSize=(25,25))
    if len(smiles) > 0:
        return "happy", 0.95
    h, w = face_gray.shape[:2]
    mouth = face_gray[int(h*0.55):min(h,int(h*0.9)), int(w*0.2):int(w*0.8)]
    if mouth.size==0:
        return "neutral", 0.5
    avg = np.mean(mouth)/255.0
    if avg < 0.38:
        return "sad", 0.6
    return "neutral", 0.5

# Heuristic age: face height ratio -> approximate
def heuristic_age(face_h, frame_h):
    ratio = float(face_h) / frame_h
    r = np.clip((ratio - 0.05) / (0.55), 0.0, 1.0)
    est = int(60 - r*52)
    est = max(5, min(90, est))
    if est < 14: rng = "child"
    elif est < 25: rng = "young"
    elif est < 50: rng = "adult"
    else: rng = "senior"
    return est, rng

# Smile intensity (mouth aspect ratio)
def mouth_aspect_ratio(gray_face):
    # Very rough: compute contrast/variation in lower part vs upper
    h = gray_face.shape[0]
    lower = gray_face[int(h*0.6):, :]
    if lower.size==0:
        return 0.0
    return float(np.std(lower))/255.0

# Head pose using solvePnP with approximate 2D landmarks from eyes/nose
def estimate_head_pose(landmarks_2d, frame_size):
    # landmarks_2d expected dict keys: left_eye, right_eye, nose, left_mouth, right_mouth
    try:
        # 3D model points of a generic face (approx)
        model_points = np.array([
            (0.0, 0.0, 0.0),        # nose tip
            (0.0, -63.6, -12.5),    # chin (approx)
            (-43.3, 32.7, -26.0),   # left eye left corner
            (43.3, 32.7, -26.0),    # right eye right corner
            (-28.9, -28.9, -24.1),  # left mouth corner
            (28.9, -28.9, -24.1)    # right mouth corner
        ], dtype=np.float64)

        size = frame_size
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4,1))

        image_points = np.array([
            landmarks_2d['nose'],
            (landmarks_2d['nose'][0], landmarks_2d['nose'][1]+size[0]*0.2),  # approximate chin
            landmarks_2d['left_eye'],
            landmarks_2d['right_eye'],
            landmarks_2d['left_mouth'],
            landmarks_2d['right_mouth']
        ], dtype=np.float64)

        success, rotation_vec, translation_vec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return None
        rmat, _ = cv2.Rodrigues(rotation_vec)
        # compute Euler angles
        sy = math.sqrt(rmat[0,0]*rmat[0,0] + rmat[1,0]*rmat[1,0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(rmat[2,1], rmat[2,2])
            y = math.atan2(-rmat[2,0], sy)
            z = math.atan2(rmat[1,0], rmat[0,0])
        else:
            x = math.atan2(-rmat[1,2], rmat[1,1])
            y = math.atan2(-rmat[2,0], sy)
            z = 0
        # convert to degrees
        return (math.degrees(x), math.degrees(y), math.degrees(z))
    except Exception:
        return None

# rPPG heart rate thread collects mean green channel from forehead ROI and estimates bpm
class RPPGEstimator:
    def __init__(self, buffer_size=250, fps=30):
        self.buffer_size = buffer_size
        self.fps = fps
        self.green_series = []
        self.lock = threading.Lock()
        self.bpm = None

    def add_roi(self, roi_bgr):
        g = float(np.mean(roi_bgr[:,:,1]))
        with self.lock:
            self.green_series.append(g)
            if len(self.green_series) > self.buffer_size:
                self.green_series.pop(0)

    def estimate(self):
        with self.lock:
            y = np.array(self.green_series, dtype=float)
        if len(y) < int(self.buffer_size*0.6) or not SCIPY_AVAILABLE:
            return None
        # detrend and normalize
        y = detrend(y)
        y = (y - np.mean(y)) / (np.std(y) + 1e-8)
        n = len(y)
        freqs = np.fft.rfftfreq(n, d=1.0/self.fps)
        fft = np.abs(np.fft.rfft(y))
        # only heart rate frequencies 0.7-3.5 Hz (~42-210 bpm)
        idx = np.where((freqs >= 0.7) & (freqs <= 3.5))[0]
        if idx.size == 0:
            return None
        best = idx[np.argmax(fft[idx])]
        hr = freqs[best] * 60.0
        self.bpm = int(hr)
        return self.bpm

rppg = RPPGEstimator(buffer_size=200, fps=30)

# Camera open (mac AVFoundation preferred)
def open_cam():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return cap

cap = open_cam()
if not cap.isOpened():
    print("Cannot open camera. Check permissions.")
    sys.exit(1)

# MediaPipe sessions if available
mp_pose_sess = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) if MEDIAPIPE_AVAILABLE else None
mp_hands_sess = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) if MEDIAPIPE_AVAILABLE else None
mp_face_mesh_sess = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=4, refine_landmarks=True, min_detection_confidence=0.5) if MEDIAPIPE_AVAILABLE else None

img_id = 0
fps = 0.0
last_time = time.time()

print("Press 'q' to quit, 's' to save")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_h, frame_w = frame.shape[:2]
    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # FPS
    now = time.time()
    fps = 0.9*fps + 0.1*(1.0/(now-last_time+1e-6))
    last_time = now

    # MediaPipe pose/hands detection
    if MEDIAPIPE_AVAILABLE:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if mp_pose_sess:
            pose_res = mp_pose_sess.process(rgb)
            if pose_res.pose_landmarks:
                for lm in pose_res.pose_landmarks.landmark:
                    pass  # You can draw landmarks or analyze posture here
        if mp_hands_sess:
            hands_res = mp_hands_sess.process(rgb)
            if hands_res.multi_hand_landmarks:
                for handLms in hands_res.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(display, handLms, mp_hands.HAND_CONNECTIONS)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60,60))

    for (x,y,w,h) in faces:
        # bounding boxes & ROI
        x2, y2 = x+w, y+h
        face_gray = gray[y:y2, x:x2]
        face_color = frame[y:y2, x:x2]

        # recognition
        name = "Unknown"
        if FACE_RECOG_AVAILABLE and known_encodings:
            try:
                # convert to rgb for face_recognition
                small_rgb = cv2.cvtColor(face_color, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(small_rgb)
                if encs:
                    dists = face_recognition.face_distance(known_encodings, encs[0])
                    best = np.argmin(dists)
                    if dists[best] < 0.5:
                        name = known_names[best]
            except Exception:
                pass

        # emotion
        emo_label, emo_conf = None, 0.0
        if advanced_emotion is not None:
            res = predict_emotion_advanced(face_gray)
            if res:
                emo_label, emo_conf = res
        if emo_label is None:
            emo_label, emo_conf = heuristic_emotion(face_gray, face_color)

        # age
        age_val, age_range = None, None
        if advanced_age is not None:
            a = predict_age_advanced(face_color)
            if a:
                age_val = a
        if age_val is None:
            age_val, age_range = heuristic_age(h, frame_h)

        # gender
        gender_label, gender_conf = None, None
        if advanced_gender is not None:
            res = predict_gender_advanced(face_color)
            if res:
                gender_label, gender_conf = res
        else:
            # heuristic: guess from face shape (very rough) using ratio
            ratio = w / float(h + 1e-6)
            gender_label = "male" if ratio > 0.9 else "female"
            gender_conf = 0.5

        # smile intensity
        smile_score = mouth_aspect_ratio(face_gray)

        # head pose via simple landmarks (use eye & nose & mouth approximations)
        # try to detect eyes and mouth corners
        leye = reye = nose = lm = rm = None
        eyes = eye_cascade.detectMultiScale(face_gray)
        if len(eyes) >= 1:
            # pick two biggest as eyes heuristically
            eyes_sorted = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)
            le = eyes_sorted[0]
            leye = (x + le[0] + le[2]/2, y + le[1] + le[3]/2)
            if len(eyes_sorted) > 1:
                re = eyes_sorted[1]
                reye = (x + re[0] + re[2]/2, y + re[1] + re[3]/2)
        # nose as center of face
        nose = (x + w/2, y + h/3)
        # mouth corners heuristic
        lm = (x + w*0.3, y + h*0.75)
        rm = (x + w*0.7, y + h*0.75)

        landmarks_2d = {'nose': nose, 'left_eye': leye or (x + w*0.35, y + h*0.35),
                        'right_eye': reye or (x + w*0.65, y + h*0.35),
                        'left_mouth': lm, 'right_mouth': rm}
        head_angles = estimate_head_pose(landmarks_2d, frame.shape)

        # draw face box and info
        cv2.rectangle(display, (x,y), (x2,y2), (0,200,0), 2)
        info_lines = [
            f"{name}",
            f"Emotion: {emo_label} ({emo_conf:.2f})",
            f"Age: {age_val} {('('+age_range+')' if age_range else '')}",
            f"Gender: {gender_label} ({gender_conf:.2f})",
            f"Smile: {smile_score:.2f}"
        ]
        if head_angles:
            info_lines.append(f"Head (pitch,yaw,roll): {head_angles[0]:.0f},{head_angles[1]:.0f},{head_angles[2]:.0f}")

        for i, line in enumerate(info_lines):
            cv2.putText(display, line, (x, y - 6 - i*18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)

        # rPPG: forehead ROI
        fh_y1 = y + int(0.06*h)
        fh_y2 = y + int(0.25*h)
        fh_x1 = x + int(0.2*w)
        fh_x2 = x + int(0.8*w)
        if fh_y2 > fh_y1 and fh_x2 > fh_x1:
            forehead = frame[fh_y1:fh_y2, fh_x1:fh_x2]
            rppg.add_roi(forehead)
            cv2.rectangle(display, (fh_x1, fh_y1), (fh_x2, fh_y2), (255,200,0), 1)

    # rPPG estimation periodically
    if SCIPY_AVAILABLE:
        bpm = rppg.estimate()
    else:
        bpm = None

    if bpm:
        cv2.putText(display, f"HR ~ {bpm} bpm", (10, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,200,0), 2)
    cv2.putText(display, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

    cv2.imshow("All-in-One Cam", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        fn = f"capture_{img_id:03d}.jpg"
        cv2.imwrite(fn, display)
        print("Saved", fn)
        img_id += 1

cap.release()
cv2.destroyAllWindows()