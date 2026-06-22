"""
model/train_model.py
Trains the feedforward neural network for VM classification.

Architecture:
  Input(4) → Dense(32, ReLU) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(3, Softmax)

Training:
  Optimizer  : Adam
  Loss       : sparse_categorical_crossentropy
  Epochs     : 50
  Batch size : 16
  Split      : 80/20 train/test

Outputs (saved in model/ directory):
  vm_model.h5   – trained Keras model
  scaler.pkl    – fitted MinMaxScaler

Run from the project root:
    python model/train_model.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "..", "task_dataset.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "vm_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# ── Load dataset ─────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(df)} records.")

# ── Features and labels ───────────────────────────────────────────────────────
FEATURE_COLS = ["cpu", "memory", "priority", "security"]
LABEL_COL    = "vm"

X_raw = df[FEATURE_COLS].values.astype(float)
y_raw = df[LABEL_COL].values

# ── Preprocessing ─────────────────────────────────────────────────────────────
scaler  = MinMaxScaler()
X       = scaler.fit_transform(X_raw)     # normalise to [0, 1]

encoder = LabelEncoder()
y       = encoder.fit_transform(y_raw)    # VM-0→0, VM-1→1, VM-2→2

print(f"  Classes: {list(encoder.classes_)}")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

# ── Model definition ──────────────────────────────────────────────────────────
model = Sequential([
    Dense(32, activation="relu",    input_shape=(4,)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(3,  activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ── Training ──────────────────────────────────────────────────────────────────
print("\nTraining …")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1,
)

# ── Evaluation ────────────────────────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy : {acc * 100:.2f}%")
print(f"Test loss     : {loss:.4f}")

# Per-class accuracy
preds   = np.argmax(model.predict(X_test, verbose=0), axis=1)
classes = ["VM-0", "VM-1", "VM-2"]
for idx, name in enumerate(classes):
    mask     = y_test == idx
    if mask.sum() > 0:
        cls_acc = (preds[mask] == y_test[mask]).mean() * 100
        print(f"  {name} accuracy: {cls_acc:.1f}%  ({mask.sum()} samples)")

# ── Save artifacts ────────────────────────────────────────────────────────────
model.save(MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"\nSaved model  → {MODEL_PATH}")
print(f"Saved scaler → {SCALER_PATH}")
print("Done.")
