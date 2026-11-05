"""
Step 2: LSTM model train karna.

Run:  python train.py
Output: sign_model.keras, labels.json, training_curves.png,
        confusion_matrix.png, classification report (console)
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from utils import GESTURES, SEQ_LEN, FEATURES, DATA_PATH, MODEL_PATH, LABELS_PATH

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_dataset():
    X, y = [], []
    for idx, gesture in enumerate(GESTURES):
        folder = os.path.join(DATA_PATH, gesture)
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"'{folder}' nahi mila. Pehle collect_data.py chalayein.")
        for fname in os.listdir(folder):
            if fname.endswith(".npy"):
                seq = np.load(os.path.join(folder, fname))
                if seq.shape == (SEQ_LEN, FEATURES):
                    X.append(seq)
                    y.append(idx)
    return np.array(X, dtype=np.float32), np.array(y)


def augment(X, y, copies=2):
    """Simple augmentation: gaussian noise + small random scaling."""
    aug_X, aug_y = [X], [y]
    for _ in range(copies):
        noise = np.random.normal(0, 0.01, X.shape).astype(np.float32)
        scale = np.random.uniform(0.95, 1.05, (X.shape[0], 1, 1)).astype(np.float32)
        aug_X.append(X * scale + noise)
        aug_y.append(y)
    return np.concatenate(aug_X), np.concatenate(aug_y)


def build_model(num_classes):
    model = Sequential([
        Input(shape=(SEQ_LEN, FEATURES)),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    X, y = load_dataset()
    print(f"Dataset: {X.shape[0]} sequences, {len(GESTURES)} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    # Augmentation sirf training data par
    X_train, y_train = augment(X_train, y_train)

    y_train_cat = to_categorical(y_train, len(GESTURES))
    y_test_cat = to_categorical(y_test, len(GESTURES))

    model = build_model(len(GESTURES))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8,
                          min_lr=1e-5),
    ]
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=150, batch_size=32, callbacks=callbacks, verbose=1)

    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest accuracy: {acc * 100:.2f}%  |  Test loss: {loss:.4f}")

    model.save(MODEL_PATH)
    with open(LABELS_PATH, "w") as f:
        json.dump(GESTURES, f)
    print(f"Model saved -> {MODEL_PATH}")

    # ---- Evaluation ----
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=GESTURES,
                                zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=GESTURES, yticklabels=GESTURES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history.history["accuracy"], label="train")
    ax[0].plot(history.history["val_accuracy"], label="val")
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()
    ax[1].plot(history.history["loss"], label="train")
    ax[1].plot(history.history["val_loss"], label="val")
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.close()
    print("Saved: confusion_matrix.png, training_curves.png")


if __name__ == "__main__":
    main()
