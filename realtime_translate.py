"""
Step 3: Real-time sign language translation.

Run:  python realtime_translate.py
Keys: q = quit | c = sentence clear | s = sentence bol kar sunao (TTS)
"""
import json
import threading
from collections import deque, Counter
import cv2
import numpy as np
import tensorflow as tf
from utils import (SEQ_LEN, MODEL_PATH, LABELS_PATH, create_hands,
                   detect_hands, draw_landmarks, extract_keypoints,
                   hands_present)

CONF_THRESHOLD = 0.85     # minimum confidence to accept a prediction
SMOOTH_WINDOW = 8         # last N predictions must agree
NO_HAND_RESET = 15        # frames without hands -> reset buffers

try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False


def speak(text):
    if not TTS_AVAILABLE or not text:
        return

    def _run():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    threading.Thread(target=_run, daemon=True).start()


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        labels = json.load(f)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam open nahi hui.")
        return

    sequence = deque(maxlen=SEQ_LEN)
    recent_preds = deque(maxlen=SMOOTH_WINDOW)
    sentence = []
    last_word = None
    no_hand_count = 0
    current_label, current_conf = "-", 0.0

    with create_hands() as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            results = detect_hands(frame, hands)
            draw_landmarks(frame, results)

            if hands_present(results):
                no_hand_count = 0
                sequence.append(extract_keypoints(results))

                if len(sequence) == SEQ_LEN:
                    inp = np.expand_dims(np.array(sequence), axis=0)
                    probs = model.predict(inp, verbose=0)[0]
                    idx = int(np.argmax(probs))
                    current_label, current_conf = labels[idx], float(probs[idx])

                    if current_conf >= CONF_THRESHOLD:
                        recent_preds.append(idx)
                    else:
                        recent_preds.append(-1)

                    if len(recent_preds) == SMOOTH_WINDOW:
                        common, count = Counter(recent_preds).most_common(1)[0]
                        if (common != -1 and count == SMOOTH_WINDOW
                                and labels[common] != last_word):
                            last_word = labels[common]
                            sentence.append(last_word)
                            sentence = sentence[-8:]
            else:
                no_hand_count += 1
                if no_hand_count >= NO_HAND_RESET:
                    sequence.clear()
                    recent_preds.clear()
                    last_word = None
                    current_label, current_conf = "-", 0.0

            # ------------------------ UI overlay ------------------------
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 45), (30, 30, 30), -1)
            cv2.putText(frame, " ".join(sentence) or "Show a sign...",
                        (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2)

            cv2.rectangle(frame, (0, h - 70), (w, h), (30, 30, 30), -1)
            cv2.putText(frame, f"Prediction: {current_label}  "
                               f"({current_conf * 100:.0f}%)",
                        (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            cv2.putText(frame, "q: quit | c: clear | s: speak",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)

            # confidence bar
            bar_w = int(current_conf * 200)
            cv2.rectangle(frame, (w - 220, h - 50), (w - 20, h - 30),
                          (80, 80, 80), 1)
            cv2.rectangle(frame, (w - 220, h - 50),
                          (w - 220 + bar_w, h - 30), (0, 255, 0), -1)

            cv2.imshow("Sign Language Translator", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                sentence.clear()
                last_word = None
            elif key == ord("s"):
                speak(" ".join(sentence))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
