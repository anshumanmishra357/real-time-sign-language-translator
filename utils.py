"""
Shared helpers for the Real-Time Sign Language Translator.
Pipeline: Webcam -> MediaPipe Hands (landmarks) -> LSTM -> predicted word
"""
import cv2
import numpy as np
import mediapipe as mp

# ----------------------------- CONFIG ---------------------------------
# Yahan apne gestures/words change kar sakte hain (collect_data.py aur
# train.py dono yahi list use karte hain).
GESTURES = ["hello", "thanks", "yes", "no", "iloveyou", "please"]

SEQ_LEN = 30          # frames per gesture sequence
NUM_SEQUENCES = 40    # sequences recorded per gesture
FEATURES = 132        # 2 hands x (21 landmarks x 3 coords + 3 wrist position)
DATA_PATH = "data"
MODEL_PATH = "sign_model.keras"
LABELS_PATH = "labels.json"
# -----------------------------------------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def create_hands():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )


def detect_hands(frame_bgr, hands):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    return hands.process(rgb)


def draw_landmarks(frame, results):
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_lms,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )


def extract_keypoints(results):
    """
    Returns a 132-length vector: [left hand (66) | right hand (66)].
    Per hand: 63 wrist-relative, scale-normalised landmark coords + 3 wrist
    position values (keeps motion information). Missing hand -> zeros.
    """
    left = np.zeros(66, dtype=np.float32)
    right = np.zeros(66, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handed in zip(results.multi_hand_landmarks,
                                    results.multi_handedness):
            label = handed.classification[0].label  # "Left" / "Right"
            pts = np.array([[p.x, p.y, p.z] for p in hand_lms.landmark],
                           dtype=np.float32)
            wrist = pts[0].copy()
            rel = pts - wrist
            scale = np.max(np.abs(rel))
            if scale < 1e-6:
                scale = 1.0
            vec = np.concatenate([(rel / scale).flatten(), wrist])
            if label == "Left":
                left = vec
            else:
                right = vec

    return np.concatenate([left, right])


def hands_present(results):
    return bool(results.multi_hand_landmarks)
