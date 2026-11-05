# Real-Time Sign Language Translator

Real-time sign language translator built with **MediaPipe** hand landmarks and an **LSTM** network (TensorFlow/Keras). It recognizes dynamic, word-level gestures from a webcam, builds sentences from them, and can speak the result aloud.

> Deep Learning course project (UCS7802), B.Tech CSE (Data Science & AI), Shri Ramswaroop Memorial University.

## Features

- Real-time gesture recognition from a standard webcam (CPU only, no GPU needed)
- Works on **dynamic gestures**: the model reads a 30-frame sequence, not a single image
- Uses hand **landmarks instead of raw pixels**, so it is less affected by background and lighting
- Confidence threshold and prediction smoothing to avoid flickering and false words
- Sentence builder with optional offline text-to-speech
- Built-in data collection tool, so you can add your own words in minutes
- Training script produces a classification report, confusion matrix and training curves

## How it works

```
Webcam -> MediaPipe Hands (21 landmarks per hand) -> 30-frame sequence (30 x 132)
       -> 3-layer LSTM -> Softmax (word probabilities) -> Sentence + optional speech
```

1. **Landmark extraction:** MediaPipe Hands detects up to two hands and returns 21 3D landmarks for each.
2. **Feature vector (132 per frame):** for each hand, landmarks are made wrist-relative and scale-normalized (63 values) and the wrist position is kept (3 values). Two hands give 66 + 66 = 132 features. A missing hand is filled with zeros.
3. **Sequence model:** 30 consecutive frames form one gesture sample and go through an LSTM classifier.
4. **Decision logic:** a word is added to the sentence only when the last 8 predictions agree and each has at least 85% confidence.

## Model architecture

| Layer | Details |
|---|---|
| Input | 30 frames x 132 features |
| LSTM | 64 units, return sequences, + BatchNormalization + Dropout 0.3 |
| LSTM | 128 units, return sequences, + Dropout 0.3 |
| LSTM | 64 units |
| Dense | 64 units, ReLU, + Dropout 0.3 |
| Output | Dense, Softmax (one unit per gesture) |

- About 203K parameters (6 gestures)
- Adam (lr 0.001), categorical cross-entropy, batch size 32, up to 150 epochs
- EarlyStopping and ReduceLROnPlateau callbacks
- 80/20 stratified train-test split, with noise and scaling augmentation on the training set only

## Project structure

```
.
├── utils.py               # Config (gesture list), MediaPipe helpers, feature extraction
├── collect_data.py        # Record gesture sequences from the webcam
├── train.py               # Train and evaluate the LSTM model
├── realtime_translate.py  # Live translation
├── requirements.txt
└── README.md
```

## Installation

Python 3.10 to 3.12 is recommended (tested on 3.12).

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

## Usage

**1. Collect data**
```bash
python collect_data.py
```
Records 40 sequences (30 frames each) for every gesture in `utils.py`. Press `q` to stop; running it again resumes where you left off.

**2. Train the model**
```bash
python train.py
```
Saves `sign_model.keras` and `labels.json`, and generates `confusion_matrix.png` and `training_curves.png`.

**3. Run live translation**
```bash
python realtime_translate.py
```

| Key | Action |
|---|---|
| `q` | Quit |
| `c` | Clear the sentence |
| `s` | Speak the sentence (text-to-speech) |

## Adding your own words

Edit the `GESTURES` list in `utils.py`, then run `collect_data.py` and `train.py` again:

```python
GESTURES = ["hello", "thanks", "yes", "no", "iloveyou", "please"]
```

Tips for good data: use good lighting, keep both hands inside the frame, and vary angle, speed and distance a little between sequences.

## Tuning

| Setting (in `realtime_translate.py`) | Effect |
|---|---|
| `CONF_THRESHOLD` (default 0.85) | Lower it if words are missed, raise it if wrong words appear |
| `SMOOTH_WINDOW` (default 8) | Higher values reduce repeated or flickering words |
| `NO_HAND_RESET` (default 15) | Frames without a hand before the buffers reset |

## Results

After training, `train.py` prints the test accuracy and a full classification report (precision, recall, F1-score) and saves:

- `confusion_matrix.png`
- `training_curves.png`

## Limitations

- Small vocabulary and a small, single-signer dataset by default, so accuracy on other people may be lower
- Only the hands are tracked; real sign languages also use facial expressions and body posture
- Recognizes isolated words, not continuous sentences
- Fixed 30-frame window, so very slow or very fast gestures may be misclassified
- Performance drops in poor lighting or when hands overlap

## Future work

- Larger vocabulary and multi-signer datasets
- Face and body pose landmarks (MediaPipe Holistic)
- GRU or Transformer models, and continuous sign recognition
- Mobile deployment with TensorFlow Lite

## Tech stack

Python, OpenCV, MediaPipe, TensorFlow / Keras, NumPy, scikit-learn, Matplotlib, Seaborn, pyttsx3

## Author

**Anshuman Mishra**, B.Tech CSE (Data Science & AI), Shri Ramswaroop Memorial University, Barabanki

## License

Released under the MIT License. Add a `LICENSE` file when you create the repository (GitHub can generate one for you).
