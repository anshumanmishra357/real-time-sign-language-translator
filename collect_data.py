"""
Step 1: Dataset collect karna (webcam se).

Run:  python collect_data.py
Har gesture ke liye NUM_SEQUENCES sequences record hongi (30 frames each).
Keys:  q = quit
Tip: Har sequence mein thoda alag speed/angle/distance rakhein taaki model
generalise kare. Roshni achhi rakhein.
"""
import os
import cv2
import numpy as np
from utils import (GESTURES, SEQ_LEN, NUM_SEQUENCES, DATA_PATH,
                   create_hands, detect_hands, draw_landmarks,
                   extract_keypoints)


def main():
    for g in GESTURES:
        os.makedirs(os.path.join(DATA_PATH, g), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam open nahi hui. Camera index check karein.")
        return

    with create_hands() as hands:
        for gesture in GESTURES:
            existing = len(os.listdir(os.path.join(DATA_PATH, gesture)))
            if existing >= NUM_SEQUENCES:
                print(f"'{gesture}' ka data pehle se hai ({existing}), skip.")
                continue

            for seq in range(existing, NUM_SEQUENCES):
                # ---- 2 second ka countdown / ready screen ----
                for _ in range(40):
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)
                    results = detect_hands(frame, hands)
                    draw_landmarks(frame, results)
                    cv2.putText(frame, f"GET READY: {gesture.upper()}",
                                (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 200, 255), 2)
                    cv2.putText(frame, f"Sequence {seq + 1}/{NUM_SEQUENCES}",
                                (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2)
                    cv2.imshow("Data Collection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                # ---- recording ----
                window = []
                while len(window) < SEQ_LEN:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)
                    results = detect_hands(frame, hands)
                    draw_landmarks(frame, results)
                    window.append(extract_keypoints(results))
                    cv2.putText(frame, f"RECORDING: {gesture.upper()}",
                                (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                    cv2.putText(frame,
                                f"Seq {seq + 1}/{NUM_SEQUENCES}  "
                                f"Frame {len(window)}/{SEQ_LEN}",
                                (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2)
                    cv2.imshow("Data Collection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                np.save(os.path.join(DATA_PATH, gesture, f"{seq}.npy"),
                        np.array(window))

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection complete!")


if __name__ == "__main__":
    main()